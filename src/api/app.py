# Imports
# > Standard library
import asyncio
import json
import multiprocessing as mp
import os
import socket
from contextlib import asynccontextmanager, suppress
from typing import Optional

# > Third-party
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HyperConfig

# > Local
from .config import APP_CONFIG, MEGABYTE
from .logging_config import setup_logging
from .queue_manager import (
    initialize_queues,
    setup_prometheus_metrics,
    start_bridge_tasks,
    stop_bridge_tasks,
)
from .routes import create_router
from .worker_manager import (
    restart_all_workers,
    start_all_workers,
    stop_all_workers,
)

# Initialize logger before anything else tries to log
logger = setup_logging(APP_CONFIG["logging_level"])

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


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

    setup_prometheus_metrics(app.state.queues, app.state.sse_response_queues)

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

    yield  # --- application runs here ---

    # ---------------------------------------------------------------------
    # Graceful shutdown
    # ---------------------------------------------------------------------
    logger.info("Shutting down application…")
    if not app.state.stop_event.is_set():
        app.state.stop_event.set()

    await stop_bridge_tasks(app.state.bridge_tasks)
    stop_all_workers(app.state.workers, app.state.stop_event)
    logger.info("All workers have been stopped and joined.")

    if app.state.monitor_task:
        app.state.monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.monitor_task
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
    logger.info(
        "Starting memory monitor with a limit set on %s MB",
        memory_limit_bytes / MEGABYTE,
    )

    while True:
        try:
            await asyncio.sleep(check_interval)
            memory_usage = check_memory_usage()
            logger.debug(
                "Memory usage: %.2f MB / %.2f MB",
                memory_usage / MEGABYTE,
                memory_limit_bytes / MEGABYTE,
            )

            if memory_usage > memory_limit_bytes:
                if app.state.restarting:
                    logger.warning("Restart already in progress. Skipping.")
                    continue

                logger.error(
                    "Memory usage (%.2f MB) exceeded limit (%.2f MB). Restarting…",
                    memory_usage / MEGABYTE,
                    memory_limit_bytes / MEGABYTE,
                )
                app.state.restarting = True

                app.state.workers = await restart_all_workers(
                    app.state.app_config,
                    app.state.stop_event,
                    app.state.workers,
                    app.state.queues,
                    app.state.mp_ctx,
                )

                logger.info("Workers and bridges restarted successfully after OOM.")
                app.state.restarting = False

        except asyncio.CancelledError:
            logger.info("Memory monitoring task cancelled.")
            break
        except Exception as exc:  # pragma: no cover – robustness
            logger.error("Error in memory_monitor: %s", exc, exc_info=True)
            app.state.restarting = False
            await asyncio.sleep(check_interval)


# ---------------------------------------------------------------------------
# FastAPI factory
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hypercorn runner (single-process)
# ---------------------------------------------------------------------------


async def run_server(
    *,
    certfile: Optional[str] = None,
    keyfile: Optional[str] = None,
    h2c: bool = False,
):
    """Serve *app* with Hypercorn inside the current process.

    Parameters
    ----------
    certfile / keyfile : str, optional
        Enable HTTPS + ALPN HTTP/2 when both are provided.
    h2c : bool, default=False
        Serve clear-text HTTP/2 (development). Ignored when TLS is enabled.
    """
    host = APP_CONFIG["uvicorn_host"]  # keep names to avoid renaming env vars
    port = APP_CONFIG["uvicorn_port"]

    logger.info("Starting app with config:\n%s", json.dumps(APP_CONFIG, indent=2))

    # DNS sanity-check (mirrors the original Uvicorn logic)
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        logger.warning(
            "Unable to resolve hostname %s. Falling back to 127.0.0.1.", host
        )
        host = "127.0.0.1"

    cfg = HyperConfig()
    cfg.bind = [f"{host}:{port}"]
    cfg.alpn_protocols = ["h2", "http/1.1"]
    cfg.worker_class = "asyncio"  # DON’T fork
    cfg.workers = 1

    if certfile and keyfile:
        cfg.certfile = certfile
        cfg.keyfile = keyfile
    elif h2c:
        cfg.h2c = True

    logger.info(
        "Starting Hypercorn on %s (h2c=%s, tls=%s)",
        cfg.bind[0],
        cfg.h2c,
        bool(certfile),
    )

    await hypercorn_serve(create_fastapi_app(), cfg)


# ---------------------------------------------------------------------------
# Module entry-point
# ---------------------------------------------------------------------------
app = create_fastapi_app()

if __name__ == "__main__":
    import sys

    # Simple CLI:  python -m src.api.app [--h2c] [certfile keyfile]
    if "--h2c" in sys.argv:
        asyncio.run(run_server(h2c=True))
    elif len(sys.argv) == 3:
        asyncio.run(run_server(certfile=sys.argv[1], keyfile=sys.argv[2]))
    else:
        asyncio.run(run_server())
