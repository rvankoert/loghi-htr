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
from batch_predictor import batch_prediction_worker
from batch_decoder import batch_decoding_worker


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
        "ERROR": logging.ERROR,
    }

    # Set up the basic logging configuration
    logging.basicConfig(
        format="[%(processName)s] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging_levels[level],
    )

    # Configure TensorFlow logger to use the custom filter
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.addFilter(TensorFlowLogFilter())
    while tf_logger.handlers:
        tf_logger.handlers.pop()

    return logging.getLogger(__name__)


async def extract_request_data(
    image: UploadFile = File(...),
    group_id: str = Form(...),
    identifier: str = Form(...),
    model: Optional[str] = Form(None),
    whitelist: List[str] = Form([]),
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
    allowed_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = image.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: "
            f"{', '.join(allowed_extensions)}",
        )

    # Read image content
    image_content = await image.read()

    # Check if the image content is empty
    if not image_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please upload a valid image file.",
        )

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
                "value provided."
            )
        logger.warning(
            "Environment variable %s not set. Using default value: %s",
            var_name,
            default_value,
        )
        return default_value

    logger.debug("Environment variable %s set to %s", var_name, value)
    return value


def initialize_queues(max_queue_size: int) -> Dict[str, any]:
    """
    Initializes the communication queues for FastAPI (async) and multiprocessing workers (mp).

    Parameters
    ----------
    max_queue_size : int
        The maximum size of the request queue.

    Returns
    -------
    dict
        A dictionary containing references to the communication queues.
        Async queues: "AsyncRequest", "AsyncDecodedResults"
        MP queues: "MPRequest", "MPPredictedBatches", "MPFinalDecodedResults"
    """
    logger = logging.getLogger(__name__)

    # Async Queues (for FastAPI)
    logger.info("Initializing async request queue")
    async_request_queue = asyncio.Queue(maxsize=max_queue_size)
    logger.info("Async request queue size: %s", max_queue_size)
    async_decoded_results_queue = asyncio.Queue()  # For SSE

    # Multiprocessing Queues (for workers)
    mp_request_queue = mp.Queue(maxsize=max_queue_size)  # Bridge from AsyncRequest
    mp_predicted_batches_queue = (
        mp.Queue()
    )  # Output of prediction worker, input to decoding worker
    mp_final_decoded_results_queue = (
        mp.Queue()
    )  # Output of decoding worker, input to bridge to AsyncDecodedResults

    # Prometheus statistics for MP queues (asyncio.Queue doesn't have simple qsize for Gauge)
    # You might need a custom way to expose async queue size if needed, e.g., via a counter
    request_queue_size_gauge = Gauge(
        "mp_request_queue_size", "MP Request queue size (fed by async bridge)"
    )
    request_queue_size_gauge.set_function(mp_request_queue.qsize)

    predicted_batches_queue_size_gauge = Gauge(
        "mp_predicted_batches_queue_size", "MP Predicted Batches queue size"
    )
    predicted_batches_queue_size_gauge.set_function(mp_predicted_batches_queue.qsize)

    final_decoded_results_queue_size_gauge = Gauge(
        "mp_final_decoded_results_queue_size", "MP Final Decoded Results queue size"
    )
    final_decoded_results_queue_size_gauge.set_function(
        mp_final_decoded_results_queue.qsize
    )

    queues = {
        "AsyncRequest": async_request_queue,
        "AsyncDecodedResults": async_decoded_results_queue,
        "MPRequest": mp_request_queue,
        "MPPredictedBatches": mp_predicted_batches_queue,
        "MPFinalDecodedResults": mp_final_decoded_results_queue,
    }

    return queues


def start_workers(
    batch_size: int,
    output_path: str,
    gpus: str,
    base_model_dir: str,
    model_name: str,
    patience: int,
    # callback_url is now specifically for predictor's critical errors
    predictor_error_callback_url: str,
    stop_event: mp.Event,
    queues: Dict[str, any],
) -> Dict[str, mp.Process]:
    """
    Initializes and starts multiple multiprocessing workers.
    predictor_error_callback_url is for batch_predictor OOM/critical errors.
    """
    logger = logging.getLogger(__name__)

    mp_request_queue = queues["MPRequest"]
    mp_predicted_batches_queue = queues["MPPredictedBatches"]
    mp_final_decoded_results_queue = queues["MPFinalDecodedResults"]

    # Start the batch prediction process
    logger.info("Starting batch prediction process")
    prediction_process = mp.Process(
        target=batch_prediction_worker,
        args=(
            mp_request_queue,
            mp_predicted_batches_queue,
            base_model_dir,
            model_name,
            output_path,  # For error logging from predictor primarily
            stop_event,
            predictor_error_callback_url,  # For OOM errors etc. in predictor
            gpus,
            batch_size,
            patience,
        ),
        name="PredictionProcess",
        daemon=True,
    )
    prediction_process.start()

    # Start the batch decoding process
    logger.info("Starting batch decoding process")
    decoding_process = mp.Process(
        target=batch_decoding_worker,
        args=(
            mp_predicted_batches_queue,
            mp_final_decoded_results_queue,
            base_model_dir,
            model_name,
            output_path,  # For any output files if decoder writes them
            stop_event,
            # callback_url is removed from decoder; results go to SSE via queue
        ),
        name="DecodingProcess",
        daemon=True,
    )
    decoding_process.start()

    workers = {"Prediction": prediction_process, "Decoding": decoding_process}
    return workers


def stop_workers(workers: Dict[str, mp.Process], stop_event: mp.Event):
    """
    Stop all worker processes gracefully.
    """
    logger = logging.getLogger(__name__)
    logger.info("Signalling worker processes to stop")
    stop_event.set()

    for name, worker in workers.items():
        logger.info("Waiting for worker process %s (%s) to finish", name, worker.pid)
        worker.join(timeout=30)  # Add a timeout
        if worker.is_alive():
            logger.warning(
                f"Worker {name} ({worker.pid}) did not terminate gracefully. Terminating."
            )
            worker.terminate()
            worker.join()
        else:
            logger.info(f"Worker {name} ({worker.pid}) finished.")


async def restart_workers(
    batch_size: int,
    output_path: str,
    gpus: str,
    base_model_dir: str,
    model_name: str,
    patience: int,
    predictor_error_callback_url: str,  # Renamed for clarity
    stop_event: mp.Event,
    workers: Dict[str, mp.Process],
    queues: Dict[str, any],
):
    """
    Restarts worker processes.
    Note: This simplified restart might lose items in async queues if not yet bridged.
    A more robust restart would involve draining async queues or persisting them.
    """
    logger = logging.getLogger(__name__)
    logger.info("Restarting workers. Waiting for MP queues to be empty...")

    # Check if MP queues are empty
    empty_count = 0
    mp_queues_to_check = [
        queues["MPRequest"],
        queues["MPPredictedBatches"],
        queues["MPFinalDecodedResults"],
    ]

    while True:
        # Check if all relevant MP queues are empty
        if all(queue.empty() for queue in mp_queues_to_check):
            empty_count += 1
            if empty_count >= 3:
                logger.info("All MP queues are empty, proceeding with worker restart.")
                break
        else:
            empty_count = 0
            logger.info("MP Queues not yet empty. Waiting...")

        await asyncio.sleep(5)  # Check every 5 seconds

    # Stop all workers
    stop_workers(
        workers, stop_event
    )  # stop_event is already set by monitor_memory usually

    # Clear stop event to allow workers to restart
    stop_event.clear()

    # Re-create MP queues for a clean state (async queues are re-used by FastAPI app state)
    # Or, one could try to clear them. For simplicity, we re-initialize the MP part.
    # The async queues are managed by the main app's lifespan.
    # This means they are *not* re-created here, but reused.
    # If items are in async queues during restart, they will be processed by new workers.

    # Restart workers with existing async queues and new MP queues for workers
    # The `initialize_queues` in `app.py` lifespan manages the single set of queues.
    # We just need to re-start the worker processes.
    # The queues dictionary passed here should be the main app.state.queues
    new_workers = start_workers(
        batch_size,
        output_path,
        gpus,
        base_model_dir,
        model_name,
        patience,
        predictor_error_callback_url,
        stop_event,
        queues,
    )
    logger.info("Workers restarted.")
    return new_workers


# --- Bridge Tasks ---
async def bridge_async_to_mp(
    async_q: asyncio.Queue,
    mp_q: mp.Queue,
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """Bridge from an asyncio.Queue to a multiprocessing.Queue."""
    logger = logging.getLogger(__name__)
    logger.info("Starting bridge: asyncio.Queue -> mp.Queue")
    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(async_q.get(), timeout=1.0)
            if item is None:  # Sentinel for graceful shutdown
                logger.info("Bridge (async->mp) received sentinel. Exiting.")
                break
            try:
                # Run blocking mp_q.put in a thread to avoid blocking event loop
                await loop.run_in_executor(
                    None, mp_q.put, item, True, 10.0
                )  # Block with timeout
                async_q.task_done()
            except mp.Full:
                logger.warning(
                    "MP queue is full. Item not bridged. Client should retry or handle."
                )
                # Potentially re-queue to async_q or handle error
                # For now, item is dropped from bridge if MP queue is full with timeout
            except Exception as e:
                logger.error(f"Error putting item to MP queue: {e}")
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logger.info("Bridge (async->mp) cancelled.")
            break
    logger.info("Bridge (async->mp) stopped.")


async def bridge_mp_to_async(
    mp_q: mp.Queue,
    async_q: asyncio.Queue,
    stop_event: mp.Event,
    loop: asyncio.AbstractEventLoop,
):
    """Bridge from a multiprocessing.Queue to an asyncio.Queue."""
    logger = logging.getLogger(__name__)
    logger.info("Starting bridge: mp.Queue -> asyncio.Queue")
    while not stop_event.is_set():
        try:
            # Run blocking mp_q.get in a thread
            item = await loop.run_in_executor(
                None, mp_q.get, True, 1.0
            )  # Block with timeout
            if item is None:  # Sentinel for graceful shutdown
                logger.info("Bridge (mp->async) received sentinel. Exiting.")
                break
            await async_q.put(item)
        except mp.queues.Empty:  # Correct exception for mp.Queue.get(timeout)
            continue
        except asyncio.CancelledError:
            logger.info("Bridge (mp->async) cancelled.")
            break
        except Exception as e:
            logger.error(
                f"Error getting item from MP queue or putting to async queue: {e}"
            )
    logger.info("Bridge (mp->async) stopped.")
