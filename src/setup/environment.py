# Imports

# > Standard library
import logging
import os
import random
from typing import List

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# > Local dependencies
from setup.config import Config


class TensorFlowLogFilter(logging.Filter):
    """Filter to exclude specific TensorFlow logging messages.

    This filter checks each log record for specific phrases that are to be
    excluded from the logs. If any of the specified phrases are found in a log
    message, the message is excluded from the logs.
    """

    def filter(self, record):
        # Exclude logs containing the specific message
        exclude_phrases = [
            "Reduce to /job:localhost/replica:0/task:0/device:CPU:",
        ]
        return not any(phrase in record.msg for phrase in exclude_phrases)


def set_deterministic(seed: int) -> None:
    """
    Sets the environment and random seeds to ensure deterministic behavior in
    TensorFlow operations.

    Parameters
    ----------
    seed : int
        The seed value to be used for setting deterministic operations across
        various libraries.

    Notes
    -----
    This function configures the environment to enforce deterministic behavior
    in TensorFlow by setting the 'TF_DETERMINISTIC_OPS' environment variable.
    It also initializes the seeds for Python's `random`, NumPy's `np.random`,
    and TensorFlow's `tf.random` with the specified seed value. This setup
    is useful for ensuring reproducibility in experiments.
    """

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpus(gpus_config: str) -> List[tf.config.PhysicalDevice]:
    """
    Configure the TensorFlow GPU environment.

    Parameters
    ----------
    gpus_config : str
        GPU selection string ('-1' for CPU, 'all' for all GPUs, or comma-separated indices).

    Returns
    -------
    List[tf.config.PhysicalDevice]
        List of active GPU devices.
    """
    """Configure the GPU environment for TensorFlow."""
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        logging.info("Selected GPU indices from config: %s", gpus_config)
        logging.info("Available GPUs: %s", gpu_devices)

        if not gpu_devices:
            logging.info("No GPUs found. Using CPU.")
            tf.config.set_visible_devices([], "GPU")
            return []

        if gpus_config == "-1":  # CPU only
            active_gpus = []
            logging.info("Using CPU only as per configuration.")
        elif gpus_config.lower() == "all":
            active_gpus = gpu_devices
            logging.info("Using all available GPUs: %s", active_gpus)
        else:
            gpu_indices_str = gpus_config.split(",")
            chosen_gpus = []
            for idx_str in gpu_indices_str:
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(gpu_devices):
                        chosen_gpus.append(gpu_devices[idx])
                    else:
                        logging.warning(f"GPU index {idx} is out of range.")
                except ValueError:
                    logging.warning(f"Invalid GPU index: {idx_str}.")
            active_gpus = chosen_gpus
            logging.info("Using specific GPU(s): %s", active_gpus)

        tf.config.set_visible_devices(active_gpus, "GPU")

        for gpu in active_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Memory growth enabled for {gpu.name}")

        return active_gpus
    except Exception as e:
        logging.error(f"Error setting up GPU environment: {e}. Falling back to CPU.")
        tf.config.set_visible_devices([], "GPU")
        return []


def setup_environment(config: Config) -> tf.distribute.Strategy:
    """
    Sets up the environment for running the TensorFlow model, including GPU
    configuration and distribution strategy.

    Parameters
    ----------
    config : Config
        The configuration object containing the parsed arguments.

    Returns
    -------
    tf.distribute.Strategy
        The TensorFlow distribution strategy based on the provided arguments.

    Notes
    -----
    This function configures the visible GPU devices based on the 'gpu'
    argument and initializes a TensorFlow distribution strategy. It also logs
    the GPU devices being used and the precision policy (float32 or
    mixed_float16) based on the 'use_float32' and 'gpu' arguments.
    """

    # Initial setup
    logging.info("Running with config:\n%s", config)

    # Set the random seed
    if config["deterministic"]:
        set_deterministic(config["seed"])

    # Set the GPU
    active_gpus = setup_gpus(config["gpu"])

    # Initialize the strategy
    strategy = initialize_strategy(config["use_float32"], active_gpus)

    return strategy


def setup_logging() -> None:
    """
    Sets up logging configuration for the application.

    Notes
    -----
    This function initializes the Python logging module with a specific format
    and date format. It also removes the default TensorFlow logger handlers to
    prevent duplicate logging and ensures that only the custom configuration is
    used.
    """

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Remove the default Tensorflow logger handlers and use our own
    tf_logger = tf.get_logger()
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()


def initialize_strategy(
    use_float32: bool, active_gpus: list[str]
) -> tf.distribute.Strategy:
    """
    Initializes the TensorFlow distribution strategy and sets the mixed
    precision policy.

    Parameters
    ----------
    use_float32 : bool
        Flag indicating whether to use float32 precision.
    active_gpus : list[str]
        A list of active GPU devices.

    Returns
    -------
    tf.distribute.Strategy
        The initialized TensorFlow distribution strategy.

    Notes
    -----
    This function sets up the MirroredStrategy for distributed training and
    configures the mixed precision policy based on the 'use_float32' and 'gpu'
    arguments. It uses 'mixed_float16' precision when GPUs are used and
    'use_float32' is False, otherwise, it defaults to 'float32' precision.
    """

    # Set the strategy for distributed training
    if len(active_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Detected multiple GPUs, using MirroredStrategy")
    else:
        strategy = tf.distribute.get_strategy()
        logging.info("Using default strategy for single GPU/CPU")

    # Set mixed precision policy
    if not use_float32 and len(active_gpus) > 0:
        # Check if all GPUs support mixed precision
        gpus_support_mixed_precision = bool(active_gpus)
        for device in active_gpus:
            if (
                tf.config.experimental.get_device_details(device)["compute_capability"][
                    0
                ]
                < 7
            ):
                gpus_support_mixed_precision = False

        # If all GPUs support mixed precision, enable it
        if gpus_support_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logging.info("Mixed precision set to 'mixed_float16'")
        else:
            logging.warning(
                "Not all GPUs support efficient mixed precision. Running in "
                "standard mode."
            )
    else:
        logging.info("Using float32 precision")

    return strategy
