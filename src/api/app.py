# Imports
# > Standard library
import asyncio
import multiprocessing as mp
import os
import socket
from contextlib import asynccontextmanager

import psutil

# > Local dependencies
from .config import APP_CONFIG, ERR_PORT_IN_USE, ERR_PERMISSION_DENIED, MEGABYTE
from .logging_config import setup_logging
from .queue_manager import (
    initialize_queues,
    start_bridge_tasks,
    stop_bridge_tasks,
)
from .worker_manager import (
    start_all_workers,
    stop_all_workers,
    restart_all_workers,
)
from .routes import create_router

# > Third-party dependencies
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import Config
from uvicorn.server import Server

# Initialize logger using the new setup
logger = setup_logging(APP_CONFIG["logging_level"])


def check_memory_usage() -> int:
    """Check the memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss + sum(
        child.memory_info().rss for child in process.children(recursive=True)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the FastAPI application."""
    mp_ctx = mp.get_context("spawn")
    app.state.mp_ctx = mp_ctx
    app.state.stop_event = mp_ctx.Event()  # For workers and bridges

    logger.info("Initializing queues")
    app.state.queues = initialize_queues(APP_CONFIG["max_queue_size"])
    app.state.async_request_queue = app.state.queues["AsyncRequest"]
    app.state.async_decoded_results_queue = app.state.queues["AsyncDecodedResults"]

    app.state.app_config = APP_CONFIG  # Store loaded config
    app.state.restarting = False

    current_loop = asyncio.get_running_loop()
    app.state.bridge_tasks = start_bridge_tasks(
        app.state.queues, app.state.stop_event, current_loop
    )

    app.state.workers = start_all_workers(
        app.state.app_config, app.state.stop_event, app.state.queues
    )

    app.state.monitor_task = asyncio.create_task(monitor_memory(app))

    yield  # Application is running

    logger.info("Shutting down application...")
    if not app.state.stop_event.is_set():
        app.state.stop_event.set()

    await stop_bridge_tasks(app.state.bridge_tasks)

    # stop_event is already set, stop_all_workers will join
    stop_all_workers(app.state.workers, app.state.stop_event)
    logger.info("All workers have been stopped and joined.")

    if app.state.monitor_task:
        app.state.monitor_task.cancel()
        try:
            await app.state.monitor_task
        except asyncio.CancelledError:
            logger.info("Memory monitoring task cancelled successfully.")
    logger.info("Application shutdown complete.")


async def monitor_memory(app: FastAPI):
    """Monitor memory usage and restart workers if limit is exceeded."""
    memory_limit_bytes = app.state.app_config["memory_limit_bytes"]
    check_interval = app.state.app_config["memory_check_interval_seconds"]

    while True:
        try:
            await asyncio.sleep(check_interval)
            memory_usage = check_memory_usage()
            logger.debug(
                f"Memory usage: {memory_usage / MEGABYTE:.2f} MB / "
                f"{memory_limit_bytes / MEGABYTE:.2f} MB"
            )

            if memory_usage > memory_limit_bytes:
                if app.state.restarting:
                    logger.warning("Restart already in progress. Skipping.")
                    continue

                logger.error(
                    f"Memory usage ({memory_usage / MEGABYTE:.2f} MB) "
                    f"exceeded limit of {memory_limit_bytes / MEGABYTE:.2f} MB. "
                    "Restarting workers and bridges..."
                )
                app.state.restarting = True

                # 1. Signal current workers and bridges to stop via the existing stop_event
                if not app.state.stop_event.is_set():
                    app.state.stop_event.set()

                # 2. Stop bridge tasks first
                await stop_bridge_tasks(app.state.bridge_tasks)

                # 3. Restart workers (this handles stopping old ones and starting new ones)
                # restart_all_workers clears and reuses the stop_event
                app.state.workers = await restart_all_workers(
                    app.state.app_config,
                    app.state.stop_event,  # Will be cleared and reused
                    app.state.workers,
                    app.state.queues,
                    app.state.mp_ctx,
                )

                # 4. Restart bridge tasks with the (now cleared) stop_event
                current_loop = asyncio.get_running_loop()
                app.state.bridge_tasks = start_bridge_tasks(
                    app.state.queues, app.state.stop_event, current_loop
                )

                logger.info(
                    "Workers and bridges restarted successfully after memory limit."
                )
                app.state.restarting = False

        except asyncio.CancelledError:
            logger.info("Memory monitoring task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in memory_monitor: {e}", exc_info=True)
            app.state.restarting = False  # Reset flag on error
            await asyncio.sleep(check_interval)  # Wait before retrying


def create_fastapi_app() -> FastAPI:
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
    api_router = create_router(app)
    app.include_router(api_router)
    return app


async def run_server():
    """Run the FastAPI server using Uvicorn."""
    host = APP_CONFIG["uvicorn_host"]
    port = APP_CONFIG["uvicorn_port"]

    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        logger.warning(
            f"Unable to resolve hostname: {host}. Falling back to 127.0.0.1."
        )
        host = "127.0.0.1"

    server_config = Config(
        app=create_fastapi_app(), host=host, port=port, workers=1, lifespan="on"
    )
    server = Server(config=server_config)

    try:
        await server.serve()
    except OSError as e:
        logger.error(f"Error starting server: {e}")
        if e.errno == ERR_PORT_IN_USE:
            logger.error(f"Port {port} is already in use.")
        elif e.errno == ERR_PERMISSION_DENIED:
            logger.error(f"Permission denied for port {port}. Try >1024 or sudo.")
    except Exception as e:
        logger.critical(f"Unexpected error starting server: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(run_server())
