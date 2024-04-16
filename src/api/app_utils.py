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
    """Filter to exclude specific TensorFlow logging messages.

    This filter checks each log record for specific phrases that are to be
    excluded from the logs. If any of the specified phrases are found in a log
    message, the message is excluded from the logs.
    """

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


def extract_request_data() -> Tuple[bytes, str, str, str, list]:
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
        whitelist : list of str
            List of classes to whitelist for output.

    Raises
    ------
    ValueError
        If required data (image, group_id, identifier) is missing or if
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

    # Check if the image content is empty or None
    if image_content is None or len(image_content) == 0:
        raise ValueError(
            "The uploaded image is empty. Please upload a valid image file.")

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

    whitelist = request.form.getlist('whitelist')

    return image_content, group_id, identifier, model, whitelist


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
            "Environment variable %s not set. Using default value: "
            "%s", var_name, default_value)
        return default_value

    logger.debug("Environment variable %s set to %s", var_name, value)
    return value


def start_workers(batch_size: int, max_queue_size: int,
                  output_path: str, gpus: str, model_path: str,
                  patience: int):
    """
    Initializes and starts multiple multiprocessing workers for image
    processing and prediction.

    This function sets up three main processes: image preparation,
    batch prediction, and batch decoding, each running in its own process. It
    also initializes thread-safe queues for communication between these
    processes and configures Prometheus gauges to monitor queue sizes.

    Parameters
    ----------
    batch_size : int
        The size of the batch for processing images.
    max_queue_size : int
        The maximum size of the request queue.
    output_path : str
        The path where the output results will be stored.
    gpus : str
        The GPU devices to use for computation.
    model_path : str
        The path to the machine learning model for predictions.
    patience : int
        The number of seconds to wait for an image to be ready before timing
        out.

    Returns
    -------
    dict
        A dictionary containing references to the worker processes under the
        keys "Preparation", "Prediction", and "Decoding".
    dict
        A dictionary containing references to the communication queues under
        the keys "Request", "Prepared", and "Predicted".
    """

    logger = logging.getLogger(__name__)

    # Create a thread-safe Queue
    logger.info("Initializing request queue")
    request_queue = mp.Queue(maxsize=max_queue_size//2)
    logger.info("Request queue size: %s", max_queue_size // 2)

    # Max size of prepared queue is half of the max size of request queue
    # expressed in number of batches
    max_prepared_queue_size = max_queue_size // 2 // batch_size
    prepared_queue = mp.Queue(maxsize=max_prepared_queue_size)
    logger.info("Prediction queue size: %s", max_prepared_queue_size)

    # Create a thread-safe Queue for predictions
    predicted_queue = mp.Queue()

    # Add request queue size to prometheus statistics
    request_queue_size_gauge = Gauge("request_queue_size",
                                     "Request queue size")
    request_queue_size_gauge.set_function(request_queue.qsize)
    prepared_queue_size_gauge = Gauge("prepared_queue_size",
                                      "Prepared queue size")
    prepared_queue_size_gauge.set_function(prepared_queue.qsize)
    predicted_queue_size_gauge = Gauge("predicted_queue_size",
                                       "Predicted queue size")
    predicted_queue_size_gauge.set_function(predicted_queue.qsize)

    # Start the image preparation process
    logger.info("Starting image preparation process")
    preparation_process = mp.Process(
        target=image_preparation_worker,
        args=(batch_size, request_queue,
              prepared_queue, model_path,
              patience),
        name="Image Preparation Process",
        daemon=True)
    preparation_process.start()

    # Start the batch prediction process
    logger.info("Starting batch prediction process")
    prediction_process = mp.Process(
        target=batch_prediction_worker,
        args=(prepared_queue, predicted_queue,
              output_path, model_path, gpus),
        name="Batch Prediction Process",
        daemon=True)
    prediction_process.start()

    # Start the batch decoding process
    logger.info("Starting batch decoding process")
    decoding_process = mp.Process(
        target=batch_decoding_worker,
        args=(predicted_queue, model_path, output_path),
        name="Batch Decoding Process",
        daemon=True)
    decoding_process.start()

    workers = {
        "Preparation": preparation_process,
        "Prediction": prediction_process,
        "Decoding": decoding_process
    }

    queues = {
        "Request": request_queue,
        "Prepared": prepared_queue,
        "Predicted": predicted_queue
    }

    return workers, queues
