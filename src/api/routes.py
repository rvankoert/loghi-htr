# Imports

# > Standard library
import datetime
from typing import List
from multiprocessing.queues import Full

# > Third-party dependencies
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# > Local dependencies
from app_utils import extract_request_data


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
        image: UploadFile = File(...),
        group_id: str = Form(...),
        identifier: str = Form(...),
        model: str = Form(None),
        whitelist: List[str] = Form([]),
    ):
        """
        Handle image prediction requests.

        Parameters
        ----------
        image : UploadFile
            The image file to be processed.
        group_id : str
            The group identifier.
        identifier : str
            The request identifier.
        model : str, optional
            The model to be used for prediction (default is None).
        whitelist : List[str], optional
            A list of whitelisted items (default is an empty list).

        Returns
        -------
        JSONResponse
            A JSON response indicating the status of the request.
        """
        try:
            data = await extract_request_data(
                image, group_id, identifier, model, whitelist)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            app.state.request_queue.put(data, block=False)
        except Full:
            raise HTTPException(
                status_code=429,
                detail="The server is currently processing a high volume of "
                       "requests. Please try again later."
            )

        return JSONResponse(
            status_code=202,
            content={
                "status": "Request received",
                "code": 202,
                "message": "Your request is being processed",
                "timestamp": datetime.datetime.now().isoformat(),
                "group_id": group_id,
                "identifier": identifier
            }
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
        for name, worker in app.state.workers.items():
            if not worker.is_alive():
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "unhealthy",
                        "code": 500,
                        "message": f"{name} worker is not alive",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )

        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "code": 200,
                "message": "All workers are alive",
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    @router.get("/ready")
    async def ready():
        """
        Check if the request queue is ready to accept new requests.

        Returns
        -------
        JSONResponse
            A JSON response indicating the readiness status of the request queue.
        """
        if app.state.request_queue.full():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unready",
                    "code": 503,
                    "message": "Request queue is full",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "code": 200,
                "message": "Request queue is not full",
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    @router.get("/prometheus")
    async def prometheus():
        metrics = generate_latest()
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)

    return router
