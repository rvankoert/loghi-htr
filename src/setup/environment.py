# Imports

# > Standard library
import argparse
import logging

# > Third-party dependencies
import tensorflow as tf


def setup_environment(args: argparse.Namespace) -> tf.distribute.Strategy:
    """
    Sets up the environment for running the TensorFlow model, including GPU
    configuration and distribution strategy.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace containing runtime arguments related to environment
        setup, like GPU selection and precision settings.

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
    logging.info(f"Running with args: {vars(args)}")

    # Set the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    logging.info(f"Available GPUs: {gpu_devices}")

    # Set the active GPUs depending on the 'gpu' argument
    if args.gpu == "-1":
        active_gpus = []
        logging.info("Using CPU")
    else:
        gpus = args.gpu.split(',')
        active_gpus = [gpu if str(i) in gpus else None for i,
                       gpu in enumerate(gpu_devices)]
        logging.info(f"Using GPU(s): {active_gpus}")

    tf.config.set_visible_devices(active_gpus, 'GPU')

    # Initialize the strategy
    strategy = initialize_strategy(args.use_float32, args.gpu)

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
    tf_logger = logging.getLogger('tensorflow')
    while tf_logger.handlers:
        tf_logger.handlers.pop()


def initialize_strategy(use_float32: bool,
                        gpu: str) -> tf.distribute.Strategy:
    """
    Initializes the TensorFlow distribution strategy and sets the mixed
    precision policy.

    Parameters
    ----------
    use_float32 : bool
        Flag indicating whether to use float32 precision.
    gpu : str
        A string indicating the GPU configuration. A value of "-1" indicates
        CPU-only mode.

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
    strategy = tf.distribute.MirroredStrategy()

    # Set mixed precision policy
    if not use_float32 and gpu != "-1":
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Using mixed_float16 precision")
    else:
        logging.info("Using float32 precision")

    return strategy
