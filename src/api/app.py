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
    initialize_queues,  # Updated
    restart_workers,
    setup_logging,
    start_workers,
    stop_workers,
    bridge_async_to_mp,  # New
    bridge_mp_to_async,  # New
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
    "output_path": get_env_variable(
        "LOGHI_OUTPUT_PATH"
    ),  # Used for error logs by workers and main output dir
    "max_queue_size": int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000")),
    "patience": float(get_env_variable("LOGHI_PATIENCE", "0.5")),
    "gpus": get_env_variable("LOGHI_GPUS", "0"),
    "predictor_error_callback_url": get_env_variable("LOGHI_CALLBACK_URL", "None"),
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
    mp_ctx = mp.get_context("spawn")  # Recommended for consistency across platforms
    stop_event = mp_ctx.Event()

    logger.info("Initializing queues")
    # Initializes AsyncRequest, AsyncDecodedResults, MPRequest, MPPredictedBatches, MPFinalDecodedResults
    queues = initialize_queues(config["max_queue_size"])

    app.state.queues = queues
    app.state.async_request_queue = queues["AsyncRequest"]
    app.state.async_decoded_results_queue = queues["AsyncDecodedResults"]
    app.state.stop_event = stop_event
    app.state.config = config
    app.state.restarting = False

    # Start bridge tasks
    loop = asyncio.get_running_loop()
    app.state.bridge_to_workers_task = asyncio.create_task(
        bridge_async_to_mp(
            queues["AsyncRequest"], queues["MPRequest"], stop_event, loop
        )
    )
    app.state.bridge_from_workers_task = asyncio.create_task(
        bridge_mp_to_async(
            queues["MPFinalDecodedResults"],
            queues["AsyncDecodedResults"],
            stop_event,
            loop,
        )
    )

    logger.info("Starting worker processes")
    workers = start_workers(
        config["batch_size"],
        config["output_path"],
        config["gpus"],
        config["base_model_dir"],
        config["model_name"],
        config["patience"],
        config["predictor_error_callback_url"],
        stop_event,
        queues,  # Pass all queues; start_workers will pick the MP ones it needs
    )
    app.state.workers = workers

    app.state.monitor_task = asyncio.create_task(monitor_memory(app))

    yield

    logger.info("Shutting down application...")
    stop_event.set()  # Signal all components to stop

    # Stop bridges
    logger.info("Stopping bridge tasks...")
    if app.state.bridge_to_workers_task:
        app.state.bridge_to_workers_task.cancel()
    if app.state.bridge_from_workers_task:
        app.state.bridge_from_workers_task.cancel()

    try:
        await asyncio.gather(
            app.state.bridge_to_workers_task,
            app.state.bridge_from_workers_task,
            return_exceptions=True,
        )
        logger.info("Bridge tasks stopped.")
    except asyncio.CancelledError:
        logger.info("Bridge tasks cancelled successfully during shutdown.")
    except Exception as e:
        logger.error(f"Error stopping bridge tasks: {e}")

    # Stop workers
    logger.info("Shutting down worker processes")
    # stop_event is already set, stop_workers will join
    stop_workers(app.state.workers, app.state.stop_event)
    logger.info("All workers have been stopped and joined")

    # Stop memory monitor
    logger.info("Stopping memory monitoring task")
    if app.state.monitor_task:
        app.state.monitor_task.cancel()
        try:
            await app.state.monitor_task
        except asyncio.CancelledError:
            logger.info("Memory monitoring task cancelled successfully")
    logger.info("Application shutdown complete.")


async def monitor_memory(app: FastAPI):
    """Monitor the memory usage and restart workers if limit is exceeded."""
    while True:
        try:
            await asyncio.sleep(MEMORY_CHECK_INTERVAL)  # Sleep first
            memory_usage = check_memory_usage()
            logger.debug(
                f"Current memory usage: {memory_usage / MEGABYTE:.2f}/"
                f"{MEMORY_LIMIT / MEGABYTE:.2f} MB"
            )

            if memory_usage > MEMORY_LIMIT:
                if app.state.restarting:
                    logger.warning("Restart already in progress, skipping new trigger.")
                    continue

                logger.error(
                    f"Memory usage ({memory_usage / MEGABYTE:.2f} MB) "
                    f"exceeded limit of {MEMORY_LIMIT / MEGABYTE:.2f} MB. "
                    "Restarting workers..."
                )
                app.state.restarting = True
                app.state.stop_event.set()  # Signal current workers and bridges to stop

                # Wait for bridges to stop (important before restarting workers that use their queues)
                logger.info("Waiting for bridge tasks to stop before worker restart...")
                if app.state.bridge_to_workers_task:
                    app.state.bridge_to_workers_task.cancel()
                if app.state.bridge_from_workers_task:
                    app.state.bridge_from_workers_task.cancel()
                try:
                    await asyncio.gather(
                        app.state.bridge_to_workers_task,
                        app.state.bridge_from_workers_task,
                        return_exceptions=True,
                    )
                except asyncio.CancelledError:
                    logger.info("Bridges cancelled for restart.")

                # Restart workers (restart_workers handles stopping old ones)
                app.state.workers = await restart_workers(
                    app.state.config["batch_size"],
                    app.state.config["output_path"],
                    app.state.config["gpus"],
                    app.state.config["base_model_dir"],
                    app.state.config["model_name"],
                    app.state.config["patience"],
                    app.state.config["predictor_error_callback_url"],
                    app.state.stop_event,  # Will be cleared and reused by restart_workers
                    app.state.workers,
                    app.state.queues,
                )

                # Restart bridge tasks
                logger.info("Restarting bridge tasks...")
                loop = asyncio.get_running_loop()
                app.state.bridge_to_workers_task = asyncio.create_task(
                    bridge_async_to_mp(
                        app.state.queues["AsyncRequest"],
                        app.state.queues["MPRequest"],
                        app.state.stop_event,
                        loop,
                    )
                )
                app.state.bridge_from_workers_task = asyncio.create_task(
                    bridge_mp_to_async(
                        app.state.queues["MPFinalDecodedResults"],
                        app.state.queues["AsyncDecodedResults"],
                        app.state.stop_event,
                        loop,
                    )
                )
                logger.info("Workers and bridges restarted after memory limit.")
                app.state.restarting = False

        except asyncio.CancelledError:
            logger.info("Memory monitoring task is being cancelled")
            break
        except Exception as e:
            logger.error(f"Error in memory_monitor: {e}", exc_info=True)
            # Potentially set restarting to false if an unexpected error occurs here
            # to allow next cycle to try again, or add more robust error handling.
            app.state.restarting = False  # Reset flag on error to allow retries
            await asyncio.sleep(MEMORY_CHECK_INTERVAL)  # Wait before retrying loop


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

    router = create_router(app)  # Pass app instance
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

    # Use app instance directly for uvicorn.Server
    # config = Config("app:app", host=host, port=port, workers=1, factory=False)
    # server = Server(config=config)
    # The above is for when app is a string. If it's an instance:

    server_config = Config(
        app=create_app(), host=host, port=port, workers=1, lifespan="on"
    )
    server = Server(config=server_config)

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


app = create_app()  # This is now done in run_server for uvicorn.Server

if __name__ == "__main__":
    # Set multiprocessing start method if not already set
    # 'spawn' is generally safer, especially on macOS and Windows
    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logger.info("Multiprocessing start method already set.")
    asyncio.run(run_server())
