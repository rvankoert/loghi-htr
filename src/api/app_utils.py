# Imports

# > Standard library
import logging
import multiprocessing as mp
import os
from typing import Tuple

# > Local dependencies
from batch_predictor import batch_prediction_worker
from image_preparator import image_preparation_worker
from batch_decoder import batch_decoding_worker

# > Third-party dependencies
from flask import request
from prometheus_client import Gauge


class TensorFlowLogFilter(logging.Filter):
    def filter(self, record):
        # Exclude logs containing the specific message
        exclude_phrases = [
            "Reduce to /job:localhost/replica:0/task:0/device:CPU:"
        ]
        return not any(phrase in record.msg for phrase in exclude_phrases)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging with the specified level and return a logger instance.

    Parameters
    ----------
    level : str, optional
        Desired logging level. Supported values are "DEBUG", "INFO",
        "WARNING", "ERROR". Default is "INFO".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    # Set up the basic logging configuration
    logging.basicConfig(
        format="[%(process)d] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging_levels[level],
    )

    # Get TensorFlow's logger and remove its handlers to prevent duplicate logs
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger(__name__)


def extract_request_data() -> Tuple[bytes, str, str, str]:
    """
    Extract image and other form data from the current request.

    Returns
    -------
    tuple of (bytes, str, str, str)
        image_content : bytes
            Content of the uploaded image.
        group_id : str
            ID of the group from form data.
        identifier : str
            Identifier from form data.
        model : str
            Location of the model to use for prediction.

    Raises
    ------
    ValueError
        If required data (image, group_id, identifier, model) is missing or if
        the image format is invalid.
    """

    # Extract the uploaded image
    image_file = request.files.get('image')
    if not image_file:
        raise ValueError("No image provided.")

    # Validate image format
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or image_file.filename.rsplit('.', 1)[1]\
            .lower() not in allowed_extensions:
        raise ValueError(
            "Invalid image format. Allowed formats: png, jpg, jpeg, gif")

    image_content = image_file.read()

    # Extract other form data
    group_id = request.form.get('group_id')
    if not group_id:
        raise ValueError("No group_id provided.")

    identifier = request.form.get('identifier')
    if not identifier:
        raise ValueError("No identifier provided.")

    model = request.form.get('model')
    if model:
        if not os.path.exists(model):
            raise ValueError(f"Model directory {model} does not exist.")

    return image_content, group_id, identifier, model


def get_env_variable(var_name: str, default_value: str = None) -> str:
    """
    Retrieve an environment variable's value or use a default value.

    Parameters
    ----------
    var_name : str
        The name of the environment variable.
    default_value : str, optional
        Default value to use if the environment variable is not set.
        Default is None.

    Returns
    -------
    str
        Value of the environment variable or the default value.

    Raises
    ------
    ValueError
        If the environment variable is not set and no default value is
        provided.
    """

    logger = logging.getLogger(__name__)

    value = os.environ.get(var_name)
    if value is None:
        if default_value is None:
            raise ValueError(
                f"Environment variable {var_name} not set and no default "
                "value provided.")
        logger.warning(
            f"Environment variable {var_name} not set. Using default value: "
            f"{default_value}")
        return default_value

    logger.debug(f"Environment variable {var_name} set to {value}")
    return value


def start_processes(batch_size: int, max_queue_size: int,
                    output_path: str, gpus: str, model_path: str,
                    patience: int):
    logger = logging.getLogger(__name__)

    # Create a thread-safe Queue
    logger.info("Initializing request queue")
    request_queue = mp.Queue(maxsize=max_queue_size//2)
    logger.info(f"Request queue size: {max_queue_size//2}")

    # Max size of prepared queue is half of the max size of request queue
    # expressed in number of batches
    max_prepared_queue_size = max_queue_size // 2 // batch_size
    prepared_queue = mp.Queue(maxsize=max_prepared_queue_size)
    logger.info(f"Prediction queue size: {max_prepared_queue_size}")

    # Create a thread-safe Queue for predictions
    predicted_queue = mp.Queue()

    # Add request queue size to prometheus statistics
    request_queue_size_gauge = Gauge("request_queue_size",
                                     "Request queue size")
    request_queue_size_gauge.set_function(lambda: request_queue.qsize())
    prepared_queue_size_gauge = Gauge("prepared_queue_size",
                                      "Prepared queue size")
    prepared_queue_size_gauge.set_function(lambda: prepared_queue.qsize())

    # Start the image preparation process
    logger.info("Starting image preparation process")
    preparation_process = mp.Process(
        target=image_preparation_worker,
        args=(batch_size, request_queue,
              prepared_queue, model_path,
              patience),
        name="Image Preparation Process")
    preparation_process.daemon = True
    preparation_process.start()

    # Start the batch prediction process
    logger.info("Starting batch prediction process")
    prediction_process = mp.Process(
        target=batch_prediction_worker,
        args=(prepared_queue, predicted_queue,
              output_path, model_path, gpus),
        name="Batch Prediction Process")
    prediction_process.daemon = True
    prediction_process.start()

    # Start the batch decoding process
    logger.info("Starting batch decoding process")
    decoding_process = mp.Process(
        target=batch_decoding_worker,
        args=(predicted_queue, model_path),
        name="Batch Decoding Process")
    decoding_process.daemon = True
    decoding_process.start()

    return request_queue, preparation_process, prediction_process, \
        decoding_process
