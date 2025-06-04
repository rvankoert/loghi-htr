# Imports

# > Standard library
import datetime
import logging
import asyncio
import json
from typing import List, Optional, Dict, Any

# > Local dependencies
from app_utils import extract_request_data

# > Third-party dependencies
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    JSONResponse,
    Response,
)
from sse_starlette.sse import EventSourceResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def create_router(app: FastAPI) -> APIRouter:
    """
    Create an API router with endpoints for prediction, health check, and
    readiness check.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance. (app.state will be used)

    Returns
    -------
    APIRouter
        The API router with the defined endpoints.
    """
    router = APIRouter()
    logger = logging.getLogger(__name__)

    @router.post("/predict")  # Was "/predict/stream"
    async def predict_stream_only(  # Renamed function for clarity
        request: Request,
        image: UploadFile = File(...),
        group_id: str = Form(...),
        identifier: str = Form(...),
        model: Optional[str] = Form(None),
        whitelist: List[str] = Form([]),
    ):
        """
        Handle image prediction requests and stream results back via SSE.
        This is now the primary and only prediction endpoint.
        """
        if app.state.restarting:

            async def error_stream():
                yield {
                    "event": "error",
                    "data": json.dumps(  # Need to import json if not already
                        _create_response(
                            503, "Service Unavailable", "Server restarting."
                        ).body.decode()
                    ),
                }

            return EventSourceResponse(error_stream())

        try:
            data_tuple = await extract_request_data(
                image, group_id, identifier, model, whitelist
            )
        except HTTPException as e:

            async def validation_error_stream(e):
                yield {"event": "error", "data": json.dumps(e.detail)}

            return EventSourceResponse(validation_error_stream(e))
        except ValueError as e:

            async def value_error_stream(e):
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Invalid input", "detail": str(e)}),
                }

            return EventSourceResponse(value_error_stream(e))

        try:
            await asyncio.wait_for(
                app.state.async_request_queue.put(data_tuple), timeout=10.0
            )
        except asyncio.TimeoutError:

            async def queue_timeout_stream():
                yield {
                    "event": "error",
                    "data": json.dumps(
                        _create_response(
                            503, "Queue Timeout", "Server busy, request queue timeout."
                        ).body.decode()
                    ),
                }

            return EventSourceResponse(queue_timeout_stream())

        logger.info(f"SSE Prediction request received for {group_id} - {identifier}")

        async def event_generator():
            request_key = f"{group_id}_{identifier}"  # Unique key for this request
            logger.debug(f"SSE event_generator started for {request_key}")
            yield {
                "event": "status",
                "data": json.dumps(
                    {"group_id": group_id, "identifier": identifier, "status": "queued"}
                ),
            }
            try:
                while True:  # Keep listening until the specific result is found or timeout/cancellation
                    try:
                        # Wait for a result from the shared queue with a timeout
                        # This timeout prevents the loop from blocking indefinitely if no matching result arrives
                        result_item: Dict[str, Any] = await asyncio.wait_for(
                            app.state.async_decoded_results_queue.get(),
                            timeout=300.0,  # e.g., 5 minutes timeout for a result to appear for this request
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"SSE event_generator for {request_key} timed out waiting for result."
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

                    # Check if this result belongs to *this* request
                    if (
                        result_item.get("group_id") == group_id
                        and result_item.get("identifier") == identifier
                    ):
                        logger.debug(f"SSE: Sending result for {request_key}")
                        yield {"event": "result", "data": json.dumps(result_item)}
                        yield {  # Explicitly signal completion for this request's stream
                            "event": "done",
                            "data": json.dumps(
                                {
                                    "group_id": group_id,
                                    "identifier": identifier,
                                    "status": "completed",
                                }
                            ),
                        }
                        app.state.async_decoded_results_queue.task_done()
                        return  # End this generator, closing the SSE connection for this client.
                    else:
                        # This result is for another request.
                        # CRITICAL: Put it back into the queue immediately for other listeners.
                        # This naive re-queueing can cause issues if no other listener picks it up quickly,
                        # potentially leading to out-of-order processing for other items or busy loops
                        # if the queue is always full.
                        # A better solution is a proper fan-out/broadcast mechanism.
                        # For now, this is a simple attempt to not "lose" messages for other streams.
                        logger.debug(
                            f"SSE event_generator for {request_key} got item for {result_item.get('group_id')}_{result_item.get('identifier')}, re-queueing."
                        )
                        try:
                            # Try to put back without blocking indefinitely
                            await asyncio.wait_for(
                                app.state.async_decoded_results_queue.put(result_item),
                                timeout=0.1,
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                f"SSE event_generator for {request_key} failed to re-queue item for other stream. Item potentially lost."
                            )
                        # No task_done here as it wasn't "processed" by this generator.
            except asyncio.CancelledError:
                logger.info(f"SSE stream for {request_key} cancelled by client.")
            except Exception as e:
                logger.error(
                    f"Error in SSE event_generator for {request_key}: {e}",
                    exc_info=True,
                )
                try:
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {"error": "Streaming error", "detail": str(e)}
                        ),
                    }
                except Exception:  # Handle error if client already disconnected
                    pass
            finally:
                logger.info(f"SSE event_generator for {request_key} finished.")

        return EventSourceResponse(event_generator())

    @router.get("/status")
    async def status(request: Request):  # request: Request for app.state
        """
        Endpoint to check the current status of the application.
        """
        is_restarting = getattr(app.state, "restarting", False)
        return JSONResponse(
            status_code=200 if not is_restarting else 503,
            content={
                "status": "restarting" if is_restarting else "running",
                "timestamp": datetime.datetime.now().isoformat(),
                "code": 200 if not is_restarting else 503,
                "async_request_queue_size": app.state.async_request_queue.qsize(),
                "async_decoded_results_queue_size": app.state.async_decoded_results_queue.qsize(),
                # MP queue sizes are exposed via Prometheus
            },
        )

    @router.get("/health")
    async def health():  # app is available via closure
        """
        Check the health of the application workers.
        """
        if app.state.restarting:
            return _create_response(
                200, "healthy", "Application is restarting"
            )  # 200 to indicate check passed, status says restarting

        unhealthy_workers = []
        for name, worker_proc in app.state.workers.items():
            if not worker_proc.is_alive():
                unhealthy_workers.append(
                    f"{name} worker (PID {worker_proc.pid}) is not alive"
                )

        if app.state.bridge_to_workers_task and app.state.bridge_to_workers_task.done():
            if app.state.bridge_to_workers_task.exception():
                unhealthy_workers.append(
                    f"Bridge to workers task failed: {app.state.bridge_to_workers_task.exception()}"
                )
            else:
                unhealthy_workers.append(
                    f"Bridge to workers task is done but should be running."
                )

        if (
            app.state.bridge_from_workers_task
            and app.state.bridge_from_workers_task.done()
        ):
            if app.state.bridge_from_workers_task.exception():
                unhealthy_workers.append(
                    f"Bridge from workers task failed: {app.state.bridge_from_workers_task.exception()}"
                )
            else:
                unhealthy_workers.append(
                    f"Bridge from workers task is done but should be running."
                )

        if unhealthy_workers:
            return _create_response(500, "unhealthy", "; ".join(unhealthy_workers))

        return _create_response(
            200, "healthy", "All workers and bridge tasks are alive and running"
        )

    @router.get("/ready")
    async def ready():  # app is available via closure
        """
        Check if the request queue is ready to accept new requests.
        """
        # Check main async request queue
        if app.state.async_request_queue.full() or app.state.restarting:
            return _create_response(
                503, "unready", "Async request queue is full or app is restarting"
            )

        # Optionally, check downstream MP queues if their fullness should gate readiness
        # For example, if MPRequest queue (fed by bridge) is also full
        # mp_request_q = app.state.queues.get("MPRequest")
        # if mp_request_q and mp_request_q.full():
        #    return _create_response(503, "unready", "Internal processing queue is full")

        return _create_response(200, "ready", "Request queue is not full")

    @router.get("/prometheus")
    async def prometheus():
        metrics = generate_latest()
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)

    return router


def _create_response(
    status_code: int, status: str, message: str, extra: dict = None
) -> JSONResponse:
    """
    Create a standardized JSON response.
    """
    content = {
        "status": status,
        "code": status_code,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)
