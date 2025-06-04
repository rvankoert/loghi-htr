# Imports

# > Standard library
import asyncio
import multiprocessing as mp
import os
import socket
from contextlib import asynccontextmanager

import psutil

# > Local dependencies
from app_utils import (
    get_env_variable,
    initialize_queues,
    restart_workers,
    setup_logging,
    start_workers,
    stop_workers,
)

# > Third-party dependencies
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import create_router
from uvicorn.config import Config
from uvicorn.server import Server

# Constants
MEGABYTE = 1024 * 1024
MEMORY_CHECK_INTERVAL = 10  # seconds
SAFE_LIMIT = int(psutil.virtual_memory().total * 0.8)
DEFAULT_MEMORY_LIMIT = min(80 * 1024 * MEGABYTE, SAFE_LIMIT)  # 80GB
MEMORY_USAGE_PERCENTAGE = 0.8  # 80%

ERR_PORT_IN_USE = 98
ERR_PERMISSION_DENIED = 13

# Set up logging
logging_level = get_env_variable("LOGGING_LEVEL", "INFO")
logger = setup_logging(logging_level)

# Configuration
config = {
    "batch_size": int(get_env_variable("LOGHI_BATCH_SIZE", "256")),
    "base_model_dir": get_env_variable("LOGHI_BASE_MODEL_DIR"),
    "model_name": get_env_variable("LOGHI_MODEL_NAME"),
    "output_path": get_env_variable("LOGHI_OUTPUT_PATH"),
    "max_queue_size": int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000")),
    "patience": float(get_env_variable("LOGHI_PATIENCE", "0.5")),
    "gpus": get_env_variable("LOGHI_GPUS", "0"),
    "callback_url": get_env_variable("LOGHI_CALLBACK_URL"),
}

# Determine memory limit
memory_limit_env = os.getenv("MEMORY_LIMIT")
if memory_limit_env:
    MEMORY_LIMIT = int(int(memory_limit_env) * MEMORY_USAGE_PERCENTAGE)
else:
    MEMORY_LIMIT = DEFAULT_MEMORY_LIMIT

logger.info(f"Memory limit set to {MEMORY_LIMIT / MEGABYTE:.2f} MB")


def check_memory_usage() -> int:
    """Check the memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss + sum(
        child.memory_info().rss for child in process.children(recursive=True)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the FastAPI application."""
    stop_event = mp.Event()

    logger.info("Starting worker processes")
    queues = initialize_queues(config["max_queue_size"])
    workers = start_workers(
        config["batch_size"],
        config["output_path"],
        config["gpus"],
        config["base_model_dir"],
        config["model_name"],
        config["patience"],
        config["callback_url"],
        stop_event,
        queues,
    )

    app.state.request_queue = queues["Request"]
    app.state.stop_event = stop_event
    app.state.workers = workers
    app.state.queues = queues
    app.state.config = config
    app.state.restarting = False
    app.state.monitor_task = asyncio.create_task(monitor_memory(app))

    yield

    logger.info("Shutting down worker processes")
    stop_workers(app.state.workers, app.state.stop_event)
    logger.info("All workers have been stopped and joined")

    logger.info("Stopping memory monitoring task")
    app.state.monitor_task.cancel()
    try:
        await app.state.monitor_task
    except asyncio.CancelledError:
        logger.info("Memory monitoring task cancelled successfully")


async def monitor_memory(app: FastAPI):
    """Monitor the memory usage and restart workers if limit is exceeded."""
    while True:
        try:
            memory_usage = check_memory_usage()
            logger.debug(
                f"Current memory usage: {memory_usage / MEGABYTE:.2f}/"
                f"{MEMORY_LIMIT / MEGABYTE:.2f} MB"
            )

            if memory_usage > MEMORY_LIMIT:
                logger.error(
                    f"Memory usage ({memory_usage / MEGABYTE:.2f} MB) "
                    f"exceeded limit of {MEMORY_LIMIT / MEGABYTE:.2f} MB. "
                    "Restarting workers..."
                )
                app.state.restarting = True
                app.state.workers = await restart_workers(
                    app.state.config["batch_size"],
                    app.state.config["output_path"],
                    app.state.config["gpus"],
                    app.state.config["base_model_dir"],
                    app.state.config["model_name"],
                    app.state.config["patience"],
                    app.state.config["callback_url"],
                    app.state.stop_event,
                    app.state.workers,
                    app.state.queues,
                )
                app.state.restarting = False

            await asyncio.sleep(MEMORY_CHECK_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Memory monitoring task is being cancelled")
            break


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Loghi-HTR API", description="API for Loghi-HTR", lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = create_router(app)
    app.include_router(router)

    return app


async def run_server():
    """Run the FastAPI server."""
    host = get_env_variable("UVICORN_HOST", "127.0.0.1")
    port = int(get_env_variable("UVICORN_PORT", "5000"))

    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        logger.error(f"Unable to resolve hostname: {host}. Falling back to localhost.")
        host = "127.0.0.1"

    config = Config("app:app", host=host, port=port, workers=1)
    server = Server(config=config)

    try:
        await server.serve()
    except OSError as e:
        logger.error(f"Error starting server: {e}")
        if e.errno == ERR_PORT_IN_USE:
            logger.error(f"Port {port} is already in use. Try a different port.")
        elif e.errno == ERR_PERMISSION_DENIED:
            logger.error(
                f"Permission denied when trying to bind to port {port}. Try a "
                "port number > 1024 or run with sudo."
            )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


app = create_app()

if __name__ == "__main__":
    asyncio.run(run_server())
