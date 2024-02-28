# Imports

# > Standard library
import logging
import multiprocessing
import os
import sys
import time
from typing import List, Tuple

# > Third-party dependencies
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa: E402
import tensorflow as tf

# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from model.management import load_model_from_directory  # noqa: E402


def create_model(model_path: str, strategy: tf.distribute.Strategy) \
        -> tf.keras.Model:
    """
    Load a pre-trained model and create utility methods.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file.
    strategy : tf.distribute.Strategy
        Strategy for distributing the model across multiple GPUs.

    Returns
    -------
    tf.keras.Model
        model : tf.keras.Model
            Loaded pre-trained model.

    Side Effects
    ------------
    - Registers custom objects needed for the model.
    - Logs various messages regarding the model and utility initialization.
    """

    logging.info("Loading model...")
    with strategy.scope():
        try:
            model = load_model_from_directory(model_path, compile=False)
            logging.info("Model %s loaded successfully", model.name)

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                model.summary()
        except Exception as e:
            logging.error("Error loading model: %s", e)
            raise e

    return model


def setup_gpu_environment(gpus: str) -> bool:
    """
    Set up the environment for batch prediction.

    Parameters:
    -----------
    gpus : str
        IDs of GPUs to be used (comma-separated).

    Returns:
    --------
    bool
        True if all GPUs support mixed precision, otherwise False.
    """

    # Set the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    logging.info("Available GPUs: %s", gpu_devices)

    # Set the active GPUs depending on the 'gpus' argument
    if gpus == "-1":
        active_gpus = []
    elif gpus.lower() == "all":
        active_gpus = gpu_devices
    else:
        gpus = gpus.split(",")
        active_gpus = []
        for i, gpu in enumerate(gpu_devices):
            if str(i) in gpus:
                active_gpus.append(gpu)

    if active_gpus:
        logging.info("Using GPU(s): %s", active_gpus)
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, 'GPU')

    # Check if all GPUs support mixed precision
    gpus_support_mixed_precision = bool(active_gpus)
    for device in active_gpus:
        tf.config.experimental.set_memory_growth(device, True)
        if tf.config.experimental.\
                get_device_details(device)['compute_capability'][0] < 7:
            gpus_support_mixed_precision = False

    # If all GPUs support mixed precision, enable it
    if gpus_support_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.debug("Mixed precision set to 'mixed_float16'")
    else:
        logging.debug(
            "Not all GPUs support efficient mixed precision. Running in "
            "standard mode.")

    return gpus_support_mixed_precision


def batch_prediction_worker(prepared_queue: multiprocessing.Queue,
                            predicted_queue: multiprocessing.Queue,
                            output_path: str,
                            model_path: str,
                            gpus: str = '0'):
    """
    Worker process for batch prediction on images.

    This function sets up a dedicated environment for batch processing of
    images using a specified model. It continuously fetches images from the
    queue until it accumulates enough for a batch prediction or a certain
    timeout is reached.

    Parameters
    ----------
    prepared_queue : multiprocessing.Queue
        Queue from which preprocessed images are fetched.
    output_path : str
        Path where predictions should be saved.
    model_path : str
        Path to the initial model file.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Default is '0'.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing status.
    """

    logging.info("Batch prediction process started")

    # If all GPUs support mixed precision, enable it
    setup_gpu_environment(gpus)
    strategy = tf.distribute.MirroredStrategy()

    # Create the model and utilities
    model = create_model(model_path, strategy)

    old_model_path = model_path

    try:
        while True:
            batch_data = prepared_queue.get()
            model_path = batch_data[4]
            batch_id = batch_data[5]
            logging.debug("Received batch %s from prepared_queue", batch_id)

            if model_path != old_model_path:
                old_model_path = model_path
                model = create_model(model_path, strategy)
                logging.info("Model reloaded due to change in model path")

            # Make predictions on the batch
            tick = time.time()
            num_predictions = handle_batch_prediction(model, predicted_queue,
                                                      batch_data, output_path)

            logging.info("Made %s predictions in %.2f seconds",
                         num_predictions, time.time() - tick)
            logging.info("Sent batch %s (%s items) to decoding queue",
                         batch_id, num_predictions)
            logging.info("%s batches waiting on prediction",
                         prepared_queue.qsize())

    except Exception as e:
        logging.error("Error in Batch Prediction Worker process: %s", e)
        raise e


def handle_batch_prediction(model: tf.keras.Model,
                            predicted_queue: multiprocessing.Queue,
                            batch_data: Tuple[tf.Tensor, ...],
                            output_path: str) -> int:
    """
    Handle the batch prediction process.

    Parameters:
    -----------
    model : tf.keras.Model
        The loaded model for predictions.
    predicted_queue : multiprocessing.Queue
        Queue where predictions are sent.
    batch_data : Tuple[tf.Tensor, ...]
        Tuple containing batch images, groups, and identifiers.
    output_path : str
        Path where predictions should be saved.

    Returns:
    --------
    int
        Number of predictions made.
    """

    # Unpack the batch data
    batch_images, batch_groups, batch_identifiers, batch_metadata, \
        _, batch_id = batch_data
    batch_info = list(zip(batch_groups, batch_identifiers))

    try:
        encoded_predictions = safe_batch_predict(model, predicted_queue,
                                                 batch_images, batch_info,
                                                 output_path)

        # Decode the predictions
        predicted_queue.put((encoded_predictions, batch_groups,
                             batch_identifiers, batch_id, batch_metadata))

        return len(encoded_predictions)

    except Exception as e:
        failed_ids = [item_id for _, item_id in batch_info]
        logging.error("Error making predictions. Skipping batch %s:\n%s",
                      batch_id, "\n".join(failed_ids))
        logging.error(e)

        for group, item_id in batch_info:
            output_prediction_error(output_path, group, item_id, e)
        return 0


def safe_batch_predict(model: tf.keras.Model,
                       predicted_queue: multiprocessing.Queue,
                       batch_images: tf.Tensor,
                       batch_info: List[Tuple[str, str]],
                       output_path: str) -> List[str]:
    """
    Attempt to predict on a batch of images using the provided model. If a
    TensorFlow Out of Memory (OOM) error occurs, the batch is split in half and
    each half is attempted again, recursively. If an OOM error occurs with a
    batch of size 1, the offending image is logged and skipped.

    Parameters
    ----------
    model : TensorFlow model
        The model used for making predictions.
    predicted_queue : multiprocessing.Queue
        Queue where predictions are sent.
    batch_images : tf.Tensor
        A tensor of images for which predictions need to be made.
    batch_info : List of tuples
        A list of tuples containing additional information (e.g., group and
        identifier) for each image in `batch_images`.
    output_path : str
        Path where any output files should be saved.

    Returns
    -------
    List
        A list of predictions made by the model. If an image causes an OOM
        error, it is skipped, and no prediction is returned for it.
    """

    try:
        return batch_predict(model, batch_images)
    except tf.errors.ResourceExhaustedError as e:
        # If the batch size is 1 and still causing OOM, then skip the image and
        # return an empty list
        if len(batch_images) == 1:
            logging.error(
                "OOM error with single image. Skipping image %s.",
                batch_info[0][1])

            output_prediction_error(
                output_path, batch_info[0][0], batch_info[0][1], e)
            return []

        logging.warning(
            "OOM error with batch size %s. Splitting batch in half and "
            "retrying.", len(batch_images))

        # Splitting batch in half
        mid_index = len(batch_images) // 2
        first_half_images = batch_images[:mid_index]
        second_half_images = batch_images[mid_index:]
        first_half_info = batch_info[:mid_index]
        second_half_info = batch_info[mid_index:]

        # Recursive calls for each half
        first_half_predictions = safe_batch_predict(
            model, predicted_queue, first_half_images,
            first_half_info, output_path,
        )
        second_half_predictions = safe_batch_predict(
            model, predicted_queue, second_half_images,
            second_half_info, output_path
        )

        return np.concatenate((first_half_predictions,
                               second_half_predictions))
    except Exception as e:
        raise e


def batch_predict(model: tf.keras.Model,
                  images: tf.Tensor) -> List[str]:
    """
    Process a batch of images using the provided model and decode the
    predictions.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-trained model for predictions.
    images : tf.Tensor
        Tensor of images for which predictions need to be made.

    Returns
    -------
    List[str]
        List of predicted texts for each image in the batch.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing and prediction
    status.
    """

    logging.debug("Initial batch size: %s", len(images))

    logging.info("Making %s predictions...", len(images))
    encoded_predictions = model.predict_on_batch(images)
    logging.debug("Predictions made")

    return encoded_predictions


def output_prediction_error(output_path: str,
                            group_id: str,
                            identifier: str,
                            text: str):
    """
    Output an error message to a file.

    Parameters
    ----------
    output_path : str
        Base path where prediction outputs should be saved.
    group_id : str
        Group ID of the image.
    identifier : str
        Identifier of the image.
    text : str
        Error message to be saved.
    """

    output_dir = os.path.join(output_path, group_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, identifier + ".error"), "w",
              encoding="utf-8") as f:
        f.write(str(text) + "\n")
