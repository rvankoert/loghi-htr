# Imports

# > Standard Library
import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


async def sse_event_generator(
    group_id: str,
    identifier: str,
    specific_results_queue: asyncio.Queue,
    unique_request_key: str,
    sse_listeners_dict: Dict[str, asyncio.Queue],
) -> AsyncGenerator[Dict[str, str], None]:
    """
    Generator for Server-Sent Events (SSE) from a specific queue.

    Listens on a queue dedicated to the request identified by `unique_request_key`,
    and emits prediction status and results as streaming events.

    Parameters
    ----------
    group_id : str
        Group identifier associated with the prediction.
    identifier : str
        Image/document identifier.
    specific_results_queue : asyncio.Queue
        Queue holding results for this specific request.
    unique_request_key : str
        Unique key associated with this SSE stream.
    sse_listeners_dict : dict
        Dictionary of all active SSE response queues.

    Yields
    ------
    dict
        A dictionary representing an SSE event to be streamed to the client.
    """
    logger.debug(f"SSE event_generator started for {unique_request_key}")

    yield {
        "event": "status",
        "data": json.dumps(
            {"group_id": group_id, "identifier": identifier, "status": "queued"}
        ),
    }

    try:
        while True:
            try:
                result_item: Dict[str, Any] = await asyncio.wait_for(
                    specific_results_queue.get(),
                    timeout=3000.0,
                )
                logger.debug(
                    f"SSE {unique_request_key}: Dequeued item {result_item.get('group_id')}_{result_item.get('identifier')}"
                )
                specific_results_queue.task_done()

            except asyncio.TimeoutError:
                logger.warning(
                    f"SSE event_generator for {unique_request_key} timed out waiting for result from its specific queue."
                )
                yield {
                    "event": "timeout",
                    "data": json.dumps(
                        {
                            "group_id": group_id,
                            "identifier": identifier,
                            "status": "timed_out",
                        }
                    ),
                }
                return

            logger.info(f"SSE {unique_request_key}: Sending result.")
            yield {"event": "result", "data": json.dumps(result_item)}
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "group_id": group_id,
                        "identifier": identifier,
                        "status": "completed",
                    }
                ),
            }
            return

    except asyncio.CancelledError:
        logger.info(f"SSE stream for {unique_request_key} cancelled by client.")
    except Exception as e:
        logger.error(
            f"Error in SSE event_generator for {unique_request_key}: {e}", exc_info=True
        )
        try:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Streaming error", "detail": str(e)}),
            }
        except Exception:
            pass
    finally:
        logger.debug(
            f"SSE {unique_request_key}: Cleaning up response queue from sse_listeners_dict."
        )
        if unique_request_key in sse_listeners_dict:
            del sse_listeners_dict[unique_request_key]
            logger.debug(f"SSE {unique_request_key}: Removed from sse_listeners_dict.")
        else:
            logger.warning(
                f"SSE {unique_request_key}: Key not found in sse_listeners_dict during cleanup."
            )
        logger.debug(f"SSE event_generator for {unique_request_key} finished.")
