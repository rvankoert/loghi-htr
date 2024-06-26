# Imports

# > Standard library
import asyncio
from contextlib import asynccontextmanager
import socket
import multiprocessing as mp

# > Third-party dependencies
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import Config
from uvicorn.server import Server

# > Local dependencies
from app_utils import (setup_logging, get_env_variable,
                       start_workers, stop_workers)
from routes import create_router

# Set up logging
logging_level = get_env_variable("LOGGING_LEVEL", "INFO")
logger = setup_logging(logging_level)

# Get Loghi-HTR options from environment variables
logger.info("Getting Loghi-HTR options from environment variables")
batch_size = int(get_env_variable("LOGHI_BATCH_SIZE", "256"))
model_path = get_env_variable("LOGHI_MODEL_PATH")
output_path = get_env_variable("LOGHI_OUTPUT_PATH")
max_queue_size = int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000"))
patience = float(get_env_variable("LOGHI_PATIENCE", "0.5"))

# Get GPU options from environment variables
logger.info("Getting GPU options from environment variables")
gpus = get_env_variable("LOGHI_GPUS", "0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifespan of the FastAPI application.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Yields
    ------
    None
    """
    # Create a stop event
    stop_event = mp.Event()

    # Startup: Start the worker processes
    logger.info("Starting worker processes")
    workers, queues = start_workers(batch_size, max_queue_size, output_path,
                                    gpus, model_path, patience, stop_event)
    # Add request queue and stop event to the app
    app.state.request_queue = queues["Request"]
    app.state.stop_event = stop_event
    app.state.workers = workers

    yield

    # Shutdown: Stop all workers and join them
    logger.info("Shutting down worker processes")
    stop_workers(app.state.workers, app.state.stop_event)
    logger.info("All workers have been stopped and joined")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        The configured FastAPI application instance.
    """
    app = FastAPI(
        title="Loghi-HTR API",
        description="API for Loghi-HTR",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Include the router
    router = create_router(app)
    app.include_router(router)

    return app


app = create_app()


async def run_server():
    """
    Run the FastAPI server.

    Returns
    -------
    None
    """
    host = get_env_variable("UVICORN_HOST", "127.0.0.1")
    port = int(get_env_variable("UVICORN_PORT", "5000"))

    # Attempt to resolve the hostname
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        logger.error(
            f"Unable to resolve hostname: {host}. Falling back to localhost.")
        host = "127.0.0.1"

    config = Config("app:app", host=host, port=port, workers=1)
    server = Server(config=config)

    try:
        await server.serve()
    except OSError as e:
        logger.error(f"Error starting server: {e}")
        if e.errno == 98:  # Address already in use
            logger.error(
                f"Port {port} is already in use. Try a different port.")
        elif e.errno == 13:  # Permission denied
            logger.error(
                f"Permission denied when trying to bind to port {port}. Try a "
                "port number > 1024 or run with sudo.")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_server())
