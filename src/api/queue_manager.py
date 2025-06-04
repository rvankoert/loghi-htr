import asyncio
import multiprocessing as mp
import logging
from typing import Dict, Any

from prometheus_client import Gauge

logger = logging.getLogger(__name__)


def initialize_queues(max_queue_size: int) -> Dict[str, Any]:
    """
    Initializes communication queues for FastAPI (async) and multiprocessing workers (mp).
    """
    logger.info("Initializing async request queue")
    async_request_queue = asyncio.Queue(maxsize=max_queue_size)
    logger.info("Async request queue size: %s", max_queue_size)
    async_decoded_results_queue = asyncio.Queue()

    mp_request_queue = mp.Queue(maxsize=max_queue_size)
    mp_predicted_batches_queue = mp.Queue()
    mp_final_decoded_results_queue = mp.Queue()

    try:
        request_queue_size_gauge = Gauge(
            "mp_request_queue_size", "MP Request queue size"
        )
        request_queue_size_gauge.set_function(mp_request_queue.qsize)

        predicted_batches_queue_size_gauge = Gauge(
            "mp_predicted_batches_queue_size", "MP Predicted Batches queue size"
        )
        predicted_batches_queue_size_gauge.set_function(
            mp_predicted_batches_queue.qsize
        )

        final_decoded_results_queue_size_gauge = Gauge(
            "mp_final_decoded_results_queue_size", "MP Final Decoded Results queue size"
        )
        final_decoded_results_queue_size_gauge.set_function(
            mp_final_decoded_results_queue.qsize
        )
    except ValueError as e:
        logger.warning(
            f"Prometheus metrics already registered or other Gauge issue: {e}"
        )

    queues = {
        "AsyncRequest": async_request_queue,
        "AsyncDecodedResults": async_decoded_results_queue,
        "MPRequest": mp_request_queue,
        "MPPredictedBatches": mp_predicted_batches_queue,
        "MPFinalDecodedResults": mp_final_decoded_results_queue,
    }
    return queues


async def bridge_async_to_mp(
    async_q: asyncio.Queue,
    mp_q: mp.Queue,
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """Bridge from an asyncio.Queue to a multiprocessing.Queue."""
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
            except mp.queues.Full:  # Corrected from mp.Full for mp.Queue
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
    async_q: asyncio.Queue,
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """Bridge from a multiprocessing.Queue to an asyncio.Queue."""
    logger.info("Starting bridge: mp.Queue -> asyncio.Queue")
    while not stop_event.is_set():
        try:
            item = await loop.run_in_executor(None, mp_q.get, True, 1.0)
            if item is None:
                logger.info("Bridge (mp->async) received sentinel. Exiting.")
                break
            await async_q.put(item)
        except mp.queues.Empty:
            continue
        except asyncio.CancelledError:
            logger.info("Bridge (mp->async) cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in bridge (mp->async): {e}")
    logger.info("Bridge (mp->async) stopped.")


def start_bridge_tasks(
    queues: Dict[str, Any], stop_event: mp.Event, loop: asyncio.AbstractEventLoop
) -> Dict[str, asyncio.Task]:
    """Starts and returns the bridge asyncio tasks."""
    logger.info("Starting bridge tasks")
    to_workers_task = asyncio.create_task(
        bridge_async_to_mp(
            queues["AsyncRequest"], queues["MPRequest"], stop_event, loop
        )
    )
    from_workers_task = asyncio.create_task(
        bridge_mp_to_async(
            queues["MPFinalDecodedResults"],
            queues["AsyncDecodedResults"],
            stop_event,
            loop,
        )
    )
    return {"to_workers": to_workers_task, "from_workers": from_workers_task}


async def stop_bridge_tasks(bridge_tasks: Dict[str, asyncio.Task]):
    """Stops the bridge asyncio tasks."""
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
