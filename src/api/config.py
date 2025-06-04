import os
import psutil
import json
import logging

# Local import from the new utils.py
from .utils import get_env_variable

logger = logging.getLogger(__name__)

# Constants
MEGABYTE = 1024 * 1024
MEMORY_CHECK_INTERVAL = 10  # seconds
MEMORY_USAGE_PERCENTAGE = 0.8  # 80%

# Error codes
ERR_PORT_IN_USE = 98
ERR_PERMISSION_DENIED = 13

# Calculate safe and default memory limits
try:
    SAFE_LIMIT = int(psutil.virtual_memory().total * 0.8)
except Exception as e:
    logger.warning(f"Could not get total virtual memory, defaulting SAFE_LIMIT: {e}")
    SAFE_LIMIT = 80 * 1024 * MEGABYTE  # Fallback to 80GB if psutil fails

DEFAULT_MEMORY_LIMIT = min(
    80 * 1024 * MEGABYTE, SAFE_LIMIT
)  # 80GB or 80% of total, whichever is smaller

# Determine memory limit
memory_limit_env = os.getenv("MEMORY_LIMIT")
if memory_limit_env:
    try:
        MEMORY_LIMIT = int(int(memory_limit_env) * MEGABYTE * MEMORY_USAGE_PERCENTAGE)
    except ValueError:
        logger.error(
            f"Invalid MEMORY_LIMIT env variable: {memory_limit_env}. Using default."
        )
        MEMORY_LIMIT = DEFAULT_MEMORY_LIMIT
else:
    MEMORY_LIMIT = DEFAULT_MEMORY_LIMIT

logger.info(f"Memory limit set to {MEMORY_LIMIT / MEGABYTE:.2f} MB")


def load_app_config() -> dict:
    """Loads application configuration from environment variables."""
    app_conf = {
        "batch_size": int(get_env_variable("LOGHI_BATCH_SIZE", "256")),
        "base_model_dir": get_env_variable("LOGHI_BASE_MODEL_DIR"),
        "model_name": get_env_variable("LOGHI_MODEL_NAME"),
        "output_path": get_env_variable("LOGHI_OUTPUT_PATH"),
        "max_queue_size": int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000")),
        "patience": float(get_env_variable("LOGHI_PATIENCE", "0.5")),
        "gpus": get_env_variable("LOGHI_GPUS", "0"),
        "logging_level": get_env_variable("LOGGING_LEVEL", "INFO").upper(),
        "uvicorn_host": get_env_variable("UVICORN_HOST", "127.0.0.1"),
        "uvicorn_port": int(get_env_variable("UVICORN_PORT", "5000")),
        "memory_limit_bytes": MEMORY_LIMIT,
        "memory_check_interval_seconds": MEMORY_CHECK_INTERVAL,
    }
    return app_conf


# Global application config instance
APP_CONFIG = load_app_config()
