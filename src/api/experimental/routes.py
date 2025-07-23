# Imports
# > Standard library
import asyncio
import datetime
import json
import logging
import uuid
from typing import List, Optional

# > Third-party dependencies
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

# > Local dependencies
from .sse_utils import sse_event_generator
from .utils import extract_request_data

logger = logging.getLogger(__name__)


def _create_error_response(status_code: int, title: str, detail: str) -> JSONResponse:
    """
    Helper to generate a standardized JSON error response.

    Parameters
    ----------
    status_code : int
        HTTP status code for the error.
    title : str
        Title of the error.
    detail : str
        Detailed explanation of the error.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": title, "detail": detail, "code": status_code},
    )


def _create_status_response(
    status_code: int, status: str, message: str, extra: Optional[dict] = None
) -> JSONResponse:
    """
    Helper to generate a JSON status/info response with a timestamp.

    Parameters
    ----------
    status_code : int
        HTTP status code.
    status : str
        Status string (e.g., "running", "ready").
    message : str
        Human-readable status message.
    extra : dict, optional
        Additional fields to include in the response.

    Returns
    -------
    JSONResponse
        Formatted status response.
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


async def _generate_error_sse_stream(status_code: int, title: str, detail: str):
    """
    Yield a single Server-Sent Event with error information.

    Parameters
    ----------
    status_code : int
        HTTP error status code.
    title : str
        Error title.
    detail : str
        Detailed error description.

    Yields
    ------
    dict
        Event dictionary to stream via SSE.
    """
    error_content = {"error": title, "detail": detail, "code": status_code}
    yield {"event": "error", "data": json.dumps(error_content)}


def create_router(app_instance: FastAPI) -> APIRouter:
    """
    Define and return the main API router for the application.

    Parameters
    ----------
    app_instance : FastAPI
        The main FastAPI app instance to access shared state.

    Returns
    -------
    APIRouter
        Configured router with endpoints.
    """
    router = APIRouter()

    @router.post("/predict")
    async def predict_endpoint(
        request: Request,
        image: UploadFile = File(...),
        group_id: str = Form(...),
        identifier: str = Form(...),
        model: Optional[str] = Form(None),
        whitelist: List[str] = Form([]),
    ):
        """
        Accept image input and return predictions via Server-Sent Events (SSE).
        """
        request_id = str(uuid.uuid4())
        app_state = request.app.state

        if app_state.restarting:
            return EventSourceResponse(
                _generate_error_sse_stream(
                    503, "Service Unavailable", "Server restarting."
                )
            )

        try:
            request_data_tuple = await extract_request_data(
                image, group_id, identifier, model, whitelist
            )
        except HTTPException as e:
            return EventSourceResponse(
                _generate_error_sse_stream(e.status_code, "Validation Error", e.detail)
            )
        except ValueError as e:
            return EventSourceResponse(
                _generate_error_sse_stream(400, "Invalid Input", str(e))
            )

        unique_request_key = f"sse_{group_id}_{identifier}_{request_id.split('-')[0]}"
        response_queue = asyncio.Queue()
        app_state.sse_response_queues[unique_request_key] = response_queue

        request_item_tuple_for_worker = (*request_data_tuple, unique_request_key)

        try:
            await asyncio.wait_for(
                app_state.async_request_queue.put(request_item_tuple_for_worker),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            return EventSourceResponse(
                _generate_error_sse_stream(
                    429, "Queue Timeout", "Server busy, request queue full."
                )
            )

        logger.info(
            f"SSE Prediction request accepted for {group_id} - {identifier} (Request Key: {unique_request_key}, API Req ID: {request_id})"
        )

        return EventSourceResponse(
            sse_event_generator(
                group_id,
                identifier,
                response_queue,
                unique_request_key,
                app_state.sse_response_queues,
            )
        )

    @router.get("/status")
    async def get_status(request: Request):
        """
        Return application status and async queue info.
        """
        app_state = request.app.state
        is_restarting = getattr(app_state, "restarting", False)
        status_str = "restarting" if is_restarting else "running"
        status_code = 503 if is_restarting else 200

        extra_info = {
            "async_request_queue_size": app_state.async_request_queue.qsize()
            if hasattr(app_state, "async_request_queue")
            else "N/A",
            "active_sse_connections": len(app_state.sse_response_queues)
            if hasattr(app_state, "sse_response_queues")
            else "N/A",
        }
        return _create_status_response(
            status_code, status_str, f"Application is {status_str}.", extra_info
        )

    @router.get("/health")
    async def get_health(request: Request):
        """
        Perform health checks on workers and bridges.
        """
        app_state = request.app.state
        if getattr(app_state, "restarting", False):
            return _create_status_response(
                200, "healthy", "Application is restarting, health check nominal."
            )

        unhealthy_components = []
        for name, worker_proc in getattr(app_state, "workers", {}).items():
            if not worker_proc or not worker_proc.is_alive():
                unhealthy_components.append(
                    f"Worker '{name}' (PID {worker_proc.pid if worker_proc else 'N/A'}) is not alive."
                )

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
        """
        Check readiness to receive traffic (e.g., not restarting, queues not full).
        """
        app_state = request.app.state
        if getattr(app_state, "restarting", False):
            return _create_status_response(503, "unready", "Application is restarting.")
        if app_state.async_request_queue.full():
            return _create_status_response(
                503, "unready", "Async request queue is full."
            )

        return _create_status_response(
            200, "ready", "Application is ready to accept requests."
        )

    @router.get("/prometheus")
    async def prometheus_metrics():
        """
        Return Prometheus metrics endpoint content.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return router
