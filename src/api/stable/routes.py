# Imports

# > Standard library
import datetime
import logging
from multiprocessing.queues import Full
from typing import List, Optional

# > Third-party dependencies
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# > Local dependencies
from .app_utils import extract_request_data


def create_router(app: FastAPI) -> APIRouter:
    """
    Create an API router with endpoints for prediction, health check, and
    readiness check.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Returns
    -------
    APIRouter
        The API router with the defined endpoints.
    """
    router = APIRouter()

    @router.post("/predict")
    async def predict(
        request: Request,
        image: UploadFile = File(...),
        group_id: str = Form(...),
        identifier: str = Form(...),
        model: Optional[str] = Form(None),
        whitelist: List[str] = Form([]),
    ):
        """
        Handle image prediction requests.

        Parameters
        ----------
        request : Request
            The FastAPI request object.
        image : UploadFile
            The image file to be processed.
        group_id : str
            The group identifier.
        identifier : str
            The request identifier.
        model : Optional[str]
            The model to be used for prediction (default is None).
        whitelist : List[str]
            A list of whitelisted items (default is an empty list).

        Returns
        -------
        JSONResponse
            A JSON response indicating the status of the request.
        """
        if app.state.restarting:
            return _create_response(
                503,
                "Service Unavailable",
                "The server is currently restarting. Please try again later.",
            )

        try:
            data = await extract_request_data(
                image, group_id, identifier, model, whitelist
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            app.state.queues["Request"].put(data, block=False)
            app.state.status_queue.put(
                {
                    "identifier": identifier,
                    "status": "queued",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "group_id": group_id,
                }
            )
        except Full:
            raise HTTPException(
                status_code=429,
                detail="The server is currently processing a "
                "high volume of requests. Please try again "
                "later.",
            )

        logging.info(f"Request received: {group_id} - {identifier}")
        return _create_response(
            202,
            "Request received",
            "Your request is being processed",
            extra={"group_id": group_id, "identifier": identifier},
        )

    @router.get("/status")
    async def status(request: Request):
        """
        Endpoint to check the current status of the application.

        Parameters
        ----------
        request : Request
            The FastAPI request object.

        Returns
        -------
        JSONResponse
            A JSON response indicating the current status of the application.
        """
        is_restarting = getattr(request.app.state, "restarting", False)
        return JSONResponse(
            status_code=200 if not is_restarting else 503,
            content={
                "status": "restarting" if is_restarting else "running",
                "timestamp": datetime.datetime.now().isoformat(),
                "code": 200 if not is_restarting else 503,
            },
        )

    @router.get("/health")
    async def health():
        """
        Check the health of the application workers.

        Returns
        -------
        JSONResponse
            A JSON response indicating the health status of the application.
        """
        if app.state.restarting:
            return _create_response(200, "healthy", "Application is restarting")

        for name, worker in app.state.workers.items():
            if not worker.is_alive():
                return _create_response(500, "unhealthy", f"{name} worker is not alive")

        return _create_response(200, "healthy", "All workers are alive")

    @router.get("/ready")
    async def ready():
        """
        Check if the request queue is ready to accept new requests.

        Returns
        -------
        JSONResponse
            A JSON response indicating the readiness status of the request
            queue.
        """
        if app.state.queues["Request"].full() or app.state.restarting:
            return _create_response(503, "unready", "Request queue is full")

        return _create_response(200, "ready", "Request queue is not full")

    @router.get("/prometheus")
    async def prometheus():
        metrics = generate_latest()
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)

    @router.get("/status/{identifier}/")
    async def status(identifier: str):
        # look up identifier in the request queue
        while True:
            try:
                obj = app.state.status_queue.get(timeout=0.1)
            except Exception:
                break
            else:
                app.state.status_dict[obj["identifier"]] = obj

        try:
            status = app.state.status_dict[identifier]
        except KeyError:
            status = None

        if status:
            return JSONResponse(status_code=200, content=status)
        else:
            return JSONResponse(
                status_code=404,
                content={"identifier": identifier, "status": "not found"},
            )

    return router


def _create_response(
    status_code: int, status: str, message: str, extra: dict = None
) -> JSONResponse:
    """
    Create a standardized JSON response.

    Parameters
    ----------
    status_code : int
        The HTTP status code.
    status : str
        The status message.
    message : str
        The detailed message.
    extra : dict, optional
        Additional key-value pairs to include in the response.

    Returns
    -------
    JSONResponse
        A standardized JSON response.
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
