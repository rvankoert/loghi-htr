# Imports
# > Standard library
import datetime
import logging
import asyncio
import json
from typing import List, Optional

# > Local dependencies
from .utils import extract_request_data  # Assuming utils.py is in the same directory
from .sse_utils import sse_event_generator  # For the SSE logic

# > Third-party dependencies
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

logger = logging.getLogger(__name__)


def _create_error_response(status_code: int, title: str, detail: str) -> JSONResponse:
    """Helper to create a JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": title, "detail": detail, "code": status_code},
    )


def _create_status_response(
    status_code: int, status: str, message: str, extra: Optional[dict] = None
) -> JSONResponse:
    """Helper to create a standardized JSON status/info response."""
    content = {
        "status": status,
        "code": status_code,
        "message": message,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


async def _generate_error_sse_stream(status_code: int, title: str, detail: str):
    """Generates an SSE stream for sending a single error event."""
    error_content = {"error": title, "detail": detail, "code": status_code}
    yield {"event": "error", "data": json.dumps(error_content)}


def create_router(app_instance: FastAPI) -> APIRouter:
    """
    Create an API router with endpoints.
    `app_instance` is passed to access `app.state`.
    """
    router = APIRouter()

    @router.post("/predict")
    async def predict_endpoint(
        request: Request,  # FastAPI request object to access app.state
        image: UploadFile = File(...),
        group_id: str = Form(...),
        identifier: str = Form(...),
        model: Optional[str] = Form(None),
        whitelist: List[str] = Form([]),  # Example: ["line_5", "char_10"]
    ):
        """
        Handles image prediction requests and streams results back via SSE.
        """
        app_state = request.app.state  # Shortcut to app.state

        if app_state.restarting:
            return EventSourceResponse(
                _generate_error_sse_stream(
                    503, "Service Unavailable", "Server restarting."
                )
            )

        try:
            # Returns: (image_content_bytes, group_id_str, identifier_str, model_str_or_None, whitelist_list_of_str)
            request_item_tuple = await extract_request_data(
                image, group_id, identifier, model, whitelist
            )
        except HTTPException as e:
            # e.detail should already be a JSON-serializable dict or string
            return EventSourceResponse(
                _generate_error_sse_stream(e.status_code, "Validation Error", e.detail)
            )
        except (
            ValueError
        ) as e:  # Should be caught by extract_request_data as HTTPException
            return EventSourceResponse(
                _generate_error_sse_stream(400, "Invalid Input", str(e))
            )

        try:
            await asyncio.wait_for(
                app_state.async_request_queue.put(request_item_tuple), timeout=10.0
            )
        except asyncio.TimeoutError:
            return EventSourceResponse(
                _generate_error_sse_stream(
                    429, "Queue Timeout", "Server busy, request queue full."
                )
            )

        logger.info(f"SSE Prediction request accepted for {group_id} - {identifier}")
        # Pass the specific queues from app_state to the sse_event_generator
        return EventSourceResponse(
            sse_event_generator(
                group_id, identifier, app_state.async_decoded_results_queue
            )
        )

    @router.get("/status")
    async def get_status(request: Request):
        app_state = request.app.state
        is_restarting = getattr(app_state, "restarting", False)
        status_str = "restarting" if is_restarting else "running"
        status_code = 503 if is_restarting else 200

        extra_info = {
            "async_request_queue_size": app_state.async_request_queue.qsize(),
            "async_decoded_results_queue_size": app_state.async_decoded_results_queue.qsize(),
            # MP queue sizes are via Prometheus
        }
        return _create_status_response(
            status_code, status_str, f"Application is {status_str}.", extra_info
        )

    @router.get("/health")
    async def get_health(request: Request):
        app_state = request.app.state
        if getattr(app_state, "restarting", False):
            return _create_status_response(
                200, "healthy", "Application is restarting, health check nominal."
            )

        unhealthy_components = []
        # Check workers
        for name, worker_proc in getattr(app_state, "workers", {}).items():
            if not worker_proc or not worker_proc.is_alive():
                unhealthy_components.append(
                    f"Worker '{name}' (PID {worker_proc.pid if worker_proc else 'N/A'}) is not alive."
                )

        # Check bridge tasks
        bridge_tasks = getattr(app_state, "bridge_tasks", {})
        for task_name, task in bridge_tasks.items():
            if not task or task.done():
                if task and task.exception():
                    unhealthy_components.append(
                        f"Bridge task '{task_name}' failed: {task.exception()}"
                    )
                else:
                    unhealthy_components.append(
                        f"Bridge task '{task_name}' is done but should be running."
                    )

        if unhealthy_components:
            return _create_status_response(
                503, "unhealthy", "; ".join(unhealthy_components)
            )
        return _create_status_response(
            200, "healthy", "All workers and bridge tasks are alive."
        )

    @router.get("/ready")
    async def get_ready(request: Request):
        app_state = request.app.state
        if getattr(app_state, "restarting", False):
            return _create_status_response(503, "unready", "Application is restarting.")
        if app_state.async_request_queue.full():
            return _create_status_response(
                503, "unready", "Async request queue is full."
            )

        # Optionally check MPRequest queue if it's a bottleneck
        # mp_request_q = app_state.queues.get("MPRequest")
        # if mp_request_q and mp_request_q.full():
        #    return _create_status_response(503, "unready", "Internal processing queue (MPRequest) is full.")

        return _create_status_response(
            200, "ready", "Application is ready to accept requests."
        )

    @router.get("/prometheus")
    async def prometheus_metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return router
