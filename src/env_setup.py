# Imports

# > Standard library
import logging

# > Third-party dependencies
import tensorflow as tf


def setup_environment(args):
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


def setup_logging():
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


def initialize_strategy(use_float32, gpu):
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
