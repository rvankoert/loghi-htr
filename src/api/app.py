# Imports
# > Standard library
import asyncio
import multiprocessing as mp
import os
import socket
from contextlib import asynccontextmanager

# > Third-party dependencies
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn.config import Config
from uvicorn.server import Server

# > Local dependencies
from .config import APP_CONFIG, ERR_PERMISSION_DENIED, ERR_PORT_IN_USE, MEGABYTE
from .logging_config import setup_logging
from .queue_manager import (
    initialize_queues,
    start_bridge_tasks,
    stop_bridge_tasks,
)
from .routes import create_router
from .worker_manager import (
    restart_all_workers,
    start_all_workers,
    stop_all_workers,
)

# Initialize logger
logger = setup_logging(APP_CONFIG["logging_level"])


def check_memory_usage() -> int:
    """
    Check the current process and its child processes' memory usage.

    Returns
    -------
    int
        Total memory usage in bytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss + sum(
        child.memory_info().rss for child in process.children(recursive=True)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager that defines FastAPI application lifespan behavior.

    Starts all required workers, bridges, and memory monitoring when the app runs,
    and ensures graceful shutdown on termination.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """
    mp_ctx = mp.get_context("spawn")
    app.state.mp_ctx = mp_ctx
    app.state.stop_event = mp_ctx.Event()

    logger.info("Initializing queues")
    app.state.sse_response_queues = {}
    app.state.queues = initialize_queues(APP_CONFIG["max_queue_size"])
    app.state.async_request_queue = app.state.queues["AsyncRequest"]
    app.state.app_config = APP_CONFIG
    app.state.restarting = False

    current_loop = asyncio.get_running_loop()
    app.state.bridge_tasks = start_bridge_tasks(
        app.state.queues,
        app.state.sse_response_queues,
        app.state.stop_event,
        current_loop,
    )

    app.state.workers = start_all_workers(
        app.state.app_config, app.state.stop_event, app.state.queues
    )

    app.state.monitor_task = asyncio.create_task(monitor_memory(app))

    yield

    logger.info("Shutting down application...")
    if not app.state.stop_event.is_set():
        app.state.stop_event.set()

    await stop_bridge_tasks(app.state.bridge_tasks)
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
    """
    Continuously monitors memory usage and restarts workers/bridges if usage exceeds limit.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """
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

                if not app.state.stop_event.is_set():
                    app.state.stop_event.set()

                await stop_bridge_tasks(app.state.bridge_tasks)

                app.state.workers = await restart_all_workers(
                    app.state.app_config,
                    app.state.stop_event,
                    app.state.workers,
                    app.state.queues,
                    app.state.mp_ctx,
                )

                current_loop = asyncio.get_running_loop()
                app.state.bridge_tasks = start_bridge_tasks(
                    app.state.queues,
                    app.state.sse_response_queues,
                    app.state.stop_event,
                    current_loop,
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
            app.state.restarting = False
            await asyncio.sleep(check_interval)


def create_fastapi_app() -> FastAPI:
    """
    Create and configure the FastAPI app instance.

    Returns
    -------
    FastAPI
        The configured FastAPI application.
    """
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
    """
    Run the FastAPI app with Uvicorn server.

    Handles DNS resolution and gracefully handles startup errors.
    """
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


app = create_fastapi_app()

if __name__ == "__main__":
    asyncio.run(run_server())
