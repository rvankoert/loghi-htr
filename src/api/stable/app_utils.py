# > Standard library
from typing import List, Optional, Dict
import logging
import multiprocessing as mp
import os
import asyncio

# > Third-party dependencies
from fastapi import UploadFile, Form, File, HTTPException
from prometheus_client import Gauge

# > Local dependencies
from .batch_predictor import batch_prediction_worker
from .batch_decoder import batch_decoding_worker


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
            "Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence",
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
        format="[%(processName)s] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging_levels[level],
    )

    # Configure TensorFlow logger to use the custom filter
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger(__name__)


async def extract_request_data(
    image: UploadFile = File(...),
    group_id: str = Form(...),
    identifier: str = Form(...),
    model: Optional[str] = Form(None),
    whitelist: List[str] = Form([])
) -> tuple[bytes, str, str, str, list]:
    """
    Extract image and other form data from the current request.

    Parameters
    ----------
    image : UploadFile
        The uploaded image file.
    group_id : str
        ID of the group from form data.
    identifier : str
        Identifier from form data.
    model : Optional[str]
        Location of the model to use for prediction.
    whitelist : List[str]
        List of classes to whitelist for output.

    Returns
    -------
    tuple of (bytes, str, str, str, list)
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
    HTTPException
        If required data is missing or if the image format is invalid.
    """
    # Validate image format
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    file_extension = image.filename.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: "
            f"{', '.join(allowed_extensions)}")

    # Read image content
    image_content = await image.read()

    # Check if the image content is empty
    if not image_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please upload a valid image "
            "file.")

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


def initialize_queues(max_queue_size: int, max_ledger_size: int) -> Dict[str, mp.Queue]:
    """
    Initializes the communication queues for the multiprocessing workers.

    Parameters
    ----------
    max_queue_size : int
        The maximum size of the request queue.
    max_ledger_size: int
        The maximum size of the status queue.

    Returns
    -------
    dict
        A dictionary containing references to the communication queues under
        the keys "Request", "Prepared", and "Predicted".
    """
    logger = logging.getLogger(__name__)

    # Create a thread-safe Queue
    logger.info("Initializing request queue")
    request_queue = mp.Queue(maxsize=max_queue_size)
    logger.info("Request queue size: %s", max_queue_size)

    # Create a thread-safe Queue for predictions
    predicted_queue = mp.Queue()

    # Add request queue size to prometheus statistics
    request_queue_size_gauge = Gauge(
        "request_queue_size", "Request queue size")
    request_queue_size_gauge.set_function(request_queue.qsize)
    predicted_queue_size_gauge = Gauge(
        "predicted_queue_size", "Predicted queue size")
    predicted_queue_size_gauge.set_function(predicted_queue.qsize)

    status_queue = mp.Queue(maxsize=max_ledger_size)
    status_queue_size_gauge = Gauge(
        "status_queue_size", "Status queue size")
    status_queue_size_gauge.set_function(status_queue.qsize)

    queues = {
        "Request": request_queue,
        "Predicted": predicted_queue,
        "Status": status_queue
    }

    return queues


def start_workers(batch_size: int, output_path: str, gpus: str, base_model_dir: str,
                  model_name: str, patience: int, stop_event: mp.Event,
                  queues: Dict[str, mp.Queue]) -> Dict[str, mp.Process]:
    """
    Initializes and starts multiple multiprocessing workers for image
    processing and prediction using existing queues.

    Parameters
    ----------
    batch_size : int
        The size of the batch for processing images.
    output_path : str
        The path where the output results will be stored.
    gpus : str
        The GPU devices to use for computation.
    base_model_dir : str
        The path to the base machine learning model for predictions.
    model_name : str
        The path to the machine learning model for predictions relative to the
        base model path.
    patience : int
        The number of seconds to wait for an image to be ready before timing
        out.
    stop_event : mp.Event
        An event to signal workers to stop.
    queues : dict
        A dictionary containing references to the communication queues.

    Returns
    -------
    dict
        A dictionary containing references to the worker processes under the
        keys "Preparation", "Prediction", and "Decoding".
    """
    logger = logging.getLogger(__name__)

    request_queue = queues["Request"]
    predicted_queue = queues["Predicted"]

    # Start the batch prediction process
    logger.info("Starting batch prediction process")
    prediction_process = mp.Process(
        target=batch_prediction_worker,
        args=(request_queue, predicted_queue, base_model_dir,
              model_name, output_path, stop_event, gpus,
              batch_size, patience),
        name="PredictionProcess",
        daemon=True)
    prediction_process.start()

    # Start the batch decoding process
    logger.info("Starting batch decoding process")
    decoding_process = mp.Process(
        target=batch_decoding_worker,
        args=(predicted_queue, base_model_dir,
              model_name, output_path, stop_event, queues["Status"]),
        name="DecodingProcess",
        daemon=True)
    decoding_process.start()

    workers = {
        "Prediction": prediction_process,
        "Decoding": decoding_process
    }

    return workers


def stop_workers(workers: Dict[str, mp.Process], stop_event: mp.Event, timeout: int = 10):
    """
    Stop all worker processes gracefully.

    Parameters
    ----------
    workers : Dict[str, mp.Process]
        A dictionary of worker processes with worker names as keys.
    stop_event : mp.Event
        An event to signal workers to stop.
    timeout : int, optional
        Maximum time (in seconds) to wait for each worker to terminate.
    """
    logger = logging.getLogger(__name__)

    # Signal all workers to stop
    stop_event.set()

    # Wait for all workers to finish
    for worker_name, worker in workers.items():
        logger.info("Waiting for worker process %s to finish", worker_name)
        worker.join(timeout=timeout)
        if worker.is_alive():
            logger.warning("Worker process %s did not terminate. Forcing termination.", worker_name)
            worker.terminate()
            worker.join(timeout=timeout)
    logger.info("workers stopped successfully.")

async def restart_workers(batch_size: int, output_path: str, gpus: str, base_model_dir: str,
                          model_name: str, patience: int, stop_event: mp.Event,
                          workers: Dict[str, mp.Process], queues: Dict[str, mp.Queue],
                          status_queue: mp.Queue = None):
    """
    Restarts worker processes when the corresponding queues are empty.

    Parameters
    ----------
    batch_size : int
        The size of the batch for processing images.
    output_path : str
        The path where the output results will be stored.
    gpus : str
        The GPU devices to use for computation.
    model_path : str
        The path to the machine learning model for predictions.
    patience : int
        The number of seconds to wait for an image to be ready before timing
        out.
    stop_event : mp.Event
        An event to signal workers to stop.
    workers : Dict[str, mp.Process]
        A dictionary of worker processes with worker names as keys.
    queues : Dict[str, mp.Queue]
        A dictionary of queues for communication between processes.
    status_queue : mp.Queue
        A queue for status updates. Default is None.
    """
    logger = logging.getLogger(__name__)
    logger.info("Restarting workers. Waiting for all queues to be empty...")

    # Check if all queues are empty multiple times before restarting workers
    # since the workers might still be processing data while the queues are
    # empty
    empty_count = 0

    while True:
        # Check if all queues are empty
        if all(queue.empty() for queue in queues.values()):
            empty_count += 1

            if empty_count >= 3:
                logger.info("All queues are empty, restarting workers.")

                # Stop all workers
                stop_workers(workers, stop_event)

                # Clear stop event to allow workers to restart
                stop_event.clear()

                # Restart workers with existing queues
                workers = start_workers(batch_size, output_path, gpus,
                                        base_model_dir, model_name,
                                        patience, stop_event,
                                        queues, status_queue=status_queue)
                return workers
        else:
            empty_count = 0

        # Sleep for a short duration to avoid busy waiting
        await asyncio.sleep(5)
