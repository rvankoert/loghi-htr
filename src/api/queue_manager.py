# Imports

# > Standard Library
import asyncio
import logging
import multiprocessing as mp
from typing import Any, Dict

# > Third-party Dependencies
from prometheus_client import Gauge

logger = logging.getLogger(__name__)


def initialize_queues(max_queue_size: int) -> Dict[str, Any]:
    """
    Initialize communication queues for asynchronous FastAPI and multiprocessing workers.

    Parameters
    ----------
    max_queue_size : int
        Maximum size for the async and multiprocessing request queues.

    Returns
    -------
    dict
        Dictionary containing initialized queues for async and multiprocessing workflows.
    """
    logger.info("Initializing async request queue")
    async_request_queue = asyncio.Queue(maxsize=max_queue_size)
    logger.info("Async request queue size: %s", max_queue_size)

    mp_request_queue = mp.Queue(maxsize=max_queue_size)
    mp_predicted_batches_queue = mp.Queue()
    mp_final_decoded_results_queue = mp.Queue()

    queues = {
        "AsyncRequest": async_request_queue,
        "MPRequest": mp_request_queue,
        "MPPredictedBatches": mp_predicted_batches_queue,
        "MPFinalDecodedResults": mp_final_decoded_results_queue,
    }
    return queues


def setup_prometheus_metrics(
    queues: Dict[str, Any], sse_response_queues: Dict[str, asyncio.Queue]
):
    """
    Set up Prometheus gauges for various application metrics.

    This function is designed to be idempotent, so it can be called multiple
    times without causing errors (e.g., during application reloads).

    Parameters
    ----------
    queues : dict
        A dictionary containing the application's communication queues.
    sse_response_queues : dict
        A dictionary of active SSE listener queues, keyed by request ID.
    """
    logger.info("Setting up Prometheus metrics.")

    # Using a list of tuples to define metrics for easy iteration
    metric_definitions = [
        (
            "async_request_queue_size",
            "Asyncio request queue size",
            lambda: queues["AsyncRequest"].qsize(),
        ),
        (
            "mp_request_queue_size",
            "MP Request queue size",
            lambda: queues["MPRequest"].qsize(),
        ),
        (
            "mp_predicted_batches_queue_size",
            "MP Predicted Batches queue size",
            lambda: queues["MPPredictedBatches"].qsize(),
        ),
        (
            "mp_final_decoded_results_queue_size",
            "MP Final Decoded Results queue size",
            lambda: queues["MPFinalDecodedResults"].qsize(),
        ),
        (
            "active_sse_connections",
            "Number of active SSE connections",
            lambda: len(sse_response_queues),
        ),
    ]

    for name, description, func in metric_definitions:
        try:
            # The Gauge is created here. The set_function makes it a "lazy" gauge
            # that calls the function to get the value when scraped.
            gauge = Gauge(name, description)
            gauge.set_function(func)
        except ValueError as e:
            # This can happen on hot-reloads in development frameworks.
            # We log a warning but don't crash, as the metric likely exists.
            logger.warning(
                f"Could not create Prometheus gauge '{name}': {e}. "
                "It might already be registered."
            )


async def bridge_async_to_mp(
    async_q: asyncio.Queue,
    mp_q: mp.Queue,
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """
    Transfer items from an asyncio queue to a multiprocessing queue.

    Parameters
    ----------
    async_q : asyncio.Queue
        Source asyncio queue.
    mp_q : mp.Queue
        Target multiprocessing queue.
    stop_event : mp.Event
        Event to signal shutdown.
    loop : asyncio.AbstractEventLoop
        The running asyncio event loop.
    """
    logger.info("Starting bridge: asyncio.Queue -> mp.Queue")
    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(async_q.get(), timeout=1.0)
            if item is None:
                logger.info("Bridge (async->mp) received sentinel. Exiting.")
                break
            try:
                await loop.run_in_executor(None, mp_q.put, item, True, 10.0)
                async_q.task_done()
            except mp.queues.Full:
                logger.warning("MP queue is full. Item not bridged.")
            except Exception as e:
                logger.error(f"Error putting item to MP queue: {e}")
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info("Bridge (async->mp) cancelled.")
            break
    logger.info("Bridge (async->mp) stopped.")


async def bridge_mp_to_async(
    mp_q: mp.Queue,
    sse_response_queues: Dict[str, asyncio.Queue],
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """
    Transfer results from a multiprocessing queue to asyncio queues based on request ID.

    Parameters
    ----------
    mp_q : mp.Queue
        Multiprocessing queue receiving results from worker processes.
    sse_response_queues : dict
        Dictionary mapping request IDs to asyncio queues.
    stop_event : mp.Event
        Event to signal shutdown.
    loop : asyncio.AbstractEventLoop
        The running asyncio event loop.
    """
    logger.info("Starting bridge: mp.Queue -> asyncio.Queue")
    while not stop_event.is_set():
        try:
            item_tuple = await loop.run_in_executor(None, mp_q.get, True, 1.0)
            if item_tuple is None:
                logger.info("Bridge (mp->specific_async_q) received sentinel. Exiting.")
                break
            result_dict, unique_request_key_str = item_tuple
            target_queue = sse_response_queues.get(unique_request_key_str)

            if target_queue:
                try:
                    await target_queue.put(result_dict)
                    logger.debug(
                        f"Bridged result for {unique_request_key_str} to its dedicated SSE queue."
                    )
                except Exception as e_put:
                    logger.error(
                        f"Error putting item to specific SSE queue for {unique_request_key_str}: {e_put}"
                    )
            else:
                logger.warning(
                    f"SSE response queue for key {unique_request_key_str} not found. Client might have disconnected or request timed out. Result discarded."
                )

        except mp.queues.Empty:
            continue
        except asyncio.CancelledError:
            logger.info("Bridge (mp->async) cancelled.")
            logger.info("Bridge (mp->specific_async_q) cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in bridge (mp->async): {e}")
            logger.error(f"Error in bridge (mp->specific_async_q): {e}", exc_info=True)
    logger.info("Bridge (mp->specific_async_q) stopped.")


def start_bridge_tasks(
    queues: Dict[str, Any],
    sse_response_queues: Dict[str, asyncio.Queue],
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
) -> Dict[str, asyncio.Task]:
    """
    Create and start bridge tasks for inter-process and async communication.

    Parameters
    ----------
    queues : dict
        Dictionary of queues used for communication.
    sse_response_queues : dict
        Mapping of unique request keys to asyncio queues.
    stop_event : mp.Event
        Event to trigger graceful shutdown.
    loop : asyncio.AbstractEventLoop
        Running asyncio loop to use for task creation.

    Returns
    -------
    dict
        Dictionary with references to created asyncio tasks.
    """
    logger.info("Starting bridge tasks")
    to_workers_task = asyncio.create_task(
        bridge_async_to_mp(
            queues["AsyncRequest"], queues["MPRequest"], stop_event, loop
        )
    )
    from_workers_task = asyncio.create_task(
        bridge_mp_to_async(
            queues["MPFinalDecodedResults"],
            sse_response_queues,
            stop_event,
            loop,
        )
    )
    return {"to_workers": to_workers_task, "from_workers": from_workers_task}


async def stop_bridge_tasks(bridge_tasks: Dict[str, asyncio.Task]):
    """
    Cancel and await completion of bridge tasks.

    Parameters
    ----------
    bridge_tasks : dict
        Dictionary of active asyncio bridge tasks to stop.
    """
    logger.info("Stopping bridge tasks...")
    if bridge_tasks.get("to_workers"):
        bridge_tasks["to_workers"].cancel()
    if bridge_tasks.get("from_workers"):
        bridge_tasks["from_workers"].cancel()

    try:
        await asyncio.gather(
            *filter(None, bridge_tasks.values()), return_exceptions=True
        )
        logger.info("Bridge tasks stopped.")
    except asyncio.CancelledError:
        logger.info("Bridge tasks cancelled successfully during operation.")
    except Exception as e:
        logger.error(f"Error stopping bridge tasks: {e}")
