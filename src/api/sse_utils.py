import asyncio
import json
import logging
from typing import Dict, Any, AsyncGenerator

logger = logging.getLogger(__name__)


async def sse_event_generator(
    group_id: str,
    identifier: str,
    async_decoded_results_queue: asyncio.Queue,
) -> AsyncGenerator[Dict[str, str], None]:
    """
    Generates Server-Sent Events for a specific prediction request.
    Listens on a shared queue for results matching group_id and identifier.
    """
    request_key = f"{group_id}_{identifier}"
    logger.debug(f"SSE event_generator started for {request_key}")

    # Initial status message
    yield {
        "event": "status",
        "data": json.dumps(
            {"group_id": group_id, "identifier": identifier, "status": "queued"}
        ),
    }

    # Store items not for this generator to put back later
    # This is a temporary holding to avoid immediate re-queueing into the same spot
    # which can cause issues if this generator is the only active one.
    # A more robust system would use per-request queues or a pub/sub mechanism.
    items_to_requeue_at_end = []

    try:
        while True:
            try:
                result_item: Dict[str, Any] = await asyncio.wait_for(
                    async_decoded_results_queue.get(),
                    timeout=300.0,  # Timeout for receiving *any* item
                )
                logger.debug(
                    f"SSE {request_key}: Dequeued item {result_item.get('group_id')}_{result_item.get('identifier')}"
                )

            except asyncio.TimeoutError:
                logger.warning(
                    f"SSE event_generator for {request_key} timed out waiting for result from main queue."
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
                return  # End this generator

            if (
                result_item.get("group_id") == group_id
                and result_item.get("identifier") == identifier
            ):
                logger.debug(f"SSE {request_key}: Matched and sending result.")
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
                async_decoded_results_queue.task_done()  # Mark this item as processed
                return  # End this generator
            else:
                # Item is not for this request.
                # Put it back onto the main queue for other generators.
                logger.debug(
                    f"SSE {request_key}: Item for {result_item.get('group_id')}_{result_item.get('identifier')} is not mine. "
                    f"Re-queueing to main async_decoded_results_queue."
                )
                try:
                    # Add to a list to requeue later to avoid immediate self-pickup if this is the only listener
                    # OR, if there are many listeners, direct requeue might be fine.
                    # A small delay before requeueing can help.
                    # For now, let's try direct requeue. If it still causes issues,
                    # we might need a slightly more sophisticated re-queue strategy (e.g., with a small delay).
                    await async_decoded_results_queue.put(result_item)
                    # DO NOT call task_done() here, as this generator didn't fully process it for its client.
                    # The generator that *does* match it will call task_done().
                except Exception as e_requeue:
                    logger.error(
                        f"SSE {request_key}: Failed to re-queue item for other stream: {e_requeue}. Item: {result_item}. Adding to final requeue list."
                    )
                    items_to_requeue_at_end.append(
                        result_item
                    )  # Fallback: try to requeue at the very end

    except asyncio.CancelledError:
        logger.info(f"SSE stream for {request_key} cancelled by client.")
    except Exception as e:
        logger.error(
            f"Error in SSE event_generator for {request_key}: {e}", exc_info=True
        )
        try:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Streaming error", "detail": str(e)}),
            }
        except Exception:
            pass
    finally:
        # Re-queue any items that were specifically marked for requeue at the end
        # (e.g., due to errors during normal re-queueing).
        if items_to_requeue_at_end:
            logger.info(
                f"SSE {request_key}: Final re-queue of {len(items_to_requeue_at_end)} items."
            )
            for item in items_to_requeue_at_end:
                try:
                    await async_decoded_results_queue.put(item)
                except Exception as e_final_requeue:
                    logger.error(
                        f"SSE {request_key}: Critical error during final re-queue. Item lost: {item}, Error: {e_final_requeue}"
                    )
        logger.info(f"SSE event_generator for {request_key} finished.")
