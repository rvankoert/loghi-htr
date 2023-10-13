# Imports

# > Standard library
import logging
import multiprocessing
from multiprocessing.queues import Empty
import os
import sys
from typing import Callable, List, Tuple
import gc

# > Local dependencies

# > Third-party dependencies
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import mixed_precision


def batch_prediction_worker(batch_size: int,
                            prepared_queue: multiprocessing.JoinableQueue,
                            model_path: str,
                            charlist_path: str,
                            output_path: str,
                            num_channels: int,
                            gpus: str = '0'):
    """
    Worker process for batch prediction on images.

    This function sets up a dedicated environment for batch processing of
    images using a specified model. It continuously fetches images from the
    queue until it accumulates enough for a batch prediction or a certain
    timeout is reached.

    Parameters
    ----------
    batch_size : int
        Number of images to process in a batch.
    prepared_queue : multiprocessing.JoinableQueue
        Queue from which preprocessed images are fetched.
    model_path : str
        Path to the model file.
    charlist_path : str
        Path to the character list file.
    output_path : str
        Path where predictions should be saved.
    num_channels : int
        Number of channels desired for the input images (e.g., 1 for grayscale,
        3 for RGB). This is used to verify that the preparation uses the
        correct format.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Default is '0'.

    Side Effects
    ------------
    - Modifies CUDA_VISIBLE_DEVICES environment variable to control GPU
    visibility.
    - Alters the system path to enable certain imports.
    - Logs various messages regarding the batch processing status.
    """

    logger = logging.getLogger(__name__)
    logger.info("Batch Prediction Worker process started")

    # Only use the specified GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    logger.debug(f"Number of GPUs available: {len(physical_devices)}")
    if physical_devices:
        all_gpus_support_mixed_precision = True

        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            logger.debug(device)

            # Get the compute capability of the GPU
            details = tf.config.experimental.get_device_details(device)
            major = details.get('compute_capability')[0]

            # Check if the compute capability is less than 7.0
            if int(major) < 7:
                all_gpus_support_mixed_precision = False
                logger.debug(
                    f"GPU {device} does not support efficient mixed precision."
                )
                break

        # If all GPUs support mixed precision, enable it
        if all_gpus_support_mixed_precision:
            mixed_precision.set_global_policy('mixed_float16')
            logger.debug("Mixed precision set to 'mixed_float16'")
        else:
            logger.debug(
                "Not all GPUs support efficient mixed precision. Running in "
                "standard mode.")
    else:
        logger.warning("No GPUs available")

    # Add parent directory to path for imports
    current_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.dirname(current_path)

    sys.path.append(parent_path)

    from utils import decode_batch_predictions, normalize_confidence

    strategy = tf.distribute.MirroredStrategy()

    try:
        with strategy.scope():
            model, utils = create_model(
                model_path, charlist_path, num_channels)
        logger.info("Model created and utilities initialized")
    except Exception as e:
        logger.error(e)
        logger.error("Error creating model. Exiting...")
        return

    TIMEOUT_DURATION = 1
    MAX_WAIT_COUNT = 3

    total_predictions = 0

    try:
        while True:
            # The goal is to accumulate a batch of images for processing.
            # However, if there's a delay in receiving images, we don't want to
            # wait indefinitely. So, we'll use a combination of timeouts and
            # counters to decide when to process whatever images we have, even
            # if we don't have a full batch.

            batch_images = []
            wait_count = 0

            logger.debug("Waiting to accumulate images for processing")

            # Wait until we have enough images for a batch
            while len(batch_images) < batch_size:
                logger.debug(
                    f"Waiting for {batch_size - len(batch_images)} more images"
                    " to be available")

                try:
                    # Wait for TIMEOUT_DURATION seconds for an image to be
                    # available in the queue
                    prepared_data = prepared_queue.get(
                        timeout=TIMEOUT_DURATION)
                    batch_images.append(prepared_data)
                    prepared_queue.task_done()
                    wait_count = 0
                except Empty:
                    wait_count += 1
                    logger.debug("Time without new images: "
                                 f"{wait_count * TIMEOUT_DURATION} seconds")

                    # If we've waited more than the maximum allowed time
                    # (MAX_WAIT_COUNT * TIMEOUT_DURATION) and we have some
                    # images in the batch, then process those images.
                    if wait_count > MAX_WAIT_COUNT and len(batch_images) > 0:

                        # Grab any remaining images in the queue up to the
                        # batch size
                        while not prepared_queue.empty()\
                                and len(batch_images) < batch_size:
                            prepared_data = prepared_queue.get()
                            batch_images.append(prepared_data)
                            prepared_queue.task_done()

                        # Reset the wait_count and break out of the loop
                        wait_count = 0
                        break

            logger.info(
                f"Retrieved batch of size {len(batch_images)}")
            logger.debug(
                f"There are {prepared_queue.qsize()} images waiting on "
                "prediction")

            # Here, make the batch prediction
            try:
                predictions = batch_predict(
                    model, batch_images, utils, decode_batch_predictions,
                    output_path, normalize_confidence)
            except Exception as e:
                logger.error(e)
                logger.error("Error making predictions. Skipping batch.")
                logger.error("Failed batch:")
                for image in batch_images:
                    logger.error(image[2])
                predictions = []

            # Update the total number of predictions made
            total_predictions += len(predictions)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Predictions:")
                for prediction in predictions:
                    logger.debug(prediction)

            logger.info(
                f"Made {len(predictions)} predictions")
            logger.info(f"Total predictions: {total_predictions}")
            logger.info(
                f"{prepared_queue.qsize()} images waiting on prediction")

            # Clear the batch images to free up memory
            logger.debug("Clearing batch images and predictions")
            del batch_images
            del predictions
            gc.collect()

    except KeyboardInterrupt:
        logger.warning(
            "Batch Prediction Worker process interrupted. Exiting...")


def create_model(model_path: str,
                 charlist_path: str,
                 num_channels: int) -> Tuple[tf.keras.Model, object]:
    """
    Load a pre-trained model and create utility methods.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file.
    charlist_path : str
        Path to the character list file.
    num_channels : int
        Number of channels desired for the input images (e.g., 1 for grayscale,
        3 for RGB).

    Returns
    -------
    tuple of (tf.keras.Model, object)
        model : tf.keras.Model
            Loaded pre-trained model.
        utils : object
            Utility methods created from the character list.

    Side Effects
    ------------
    - Registers custom objects needed for the model.
    - Logs various messages regarding the model and utility initialization.
    """

    from model import CERMetric, WERMetric, CTCLoss
    from utils import Utils

    logger = logging.getLogger(__name__)

    # Register custom objects
    get_custom_objects().update({
        'CERMetric': CERMetric,
        'WERMetric': WERMetric,
        'CTCLoss': CTCLoss,
    })
    logger.debug("Custom objects registered")

    logger.info("Loading model...")
    model = tf.keras.saving.load_model(model_path)
    logger.info("Model loaded successfully")

    model_channels = model.input_shape[3]
    if model_channels != num_channels:
        raise ValueError(
            f"Model expects {model_channels} channels, but {num_channels} "
            "were provided")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(model.summary())

    with open(charlist_path) as file:
        charlist = list(char for char in file.read())
    utils = Utils(charlist, use_mask=True)
    logger.debug("Utilities initialized")

    return model, utils


def batch_predict(model: tf.keras.Model,
                  batch: List[Tuple[tf.Tensor, str, str]],
                  utils: object,
                  decoder: Callable,
                  output_path: str,
                  confidence_normalizer: Callable) -> List[str]:
    """
    Process a batch of images using the provided model and decode the
    predictions.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-trained model for predictions.
    batch : List[Tuple[tf.Tensor, str, str]]
        List of tuples containing images, groups, and identifiers.
    utils : object
        Utility methods for handling predictions.
    decoder : Callable
        Function to decode batch predictions.
    output_path : str
        Path where predictions should be saved.
    confidence_normalizer : Callable
        Function to normalize the confidence of the predictions.

    Returns
    -------
    List[str]
        List of predicted texts for each image in the batch.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing and prediction
    status.
    """

    logger = logging.getLogger(__name__)

    logger.debug(f"Initial batch size: {len(batch)}")

    # Unpack the batch
    images, groups, identifiers = map(list, zip(*batch))

    # Determine the maximum width of the images in the batch
    max_width = 0
    for image in images:
        if image.shape[0] > max_width:
            max_width = image.shape[0]
    logger.debug(f"Determined max width: {max_width}")

    # Pad the images to the maximum width
    for i in range(len(images)):
        images[i] = tf.image.resize_with_pad(images[i], max_width, 64)
    images = tf.stack(images)

    logger.debug(f"Batch shape after padding: {images.shape}")

    batch = tf.convert_to_tensor(images)
    logger.debug("Converted batch to tensor")

    logger.info("Making predictions...")
    encoded_predictions = model(images)
    logger.debug("Predictions made")

    # Clear the session to free up memory
    logger.debug("Clearing session...")
    tf.keras.backend.clear_session()
    logger.debug("Session cleared")

    logger.debug("Decoding predictions...")
    decoded_predictions = decoder(encoded_predictions, utils)[0]
    logger.debug("Predictions decoded")

    logger.debug("Outputting predictions...")
    predicted_texts = output_predictions(decoded_predictions,
                                         groups,
                                         identifiers,
                                         output_path,
                                         confidence_normalizer)
    logger.debug("Predictions outputted")

    return predicted_texts


def output_predictions(predictions: List[Tuple[float, str]],
                       groups: List[str],
                       identifiers: List[str],
                       output_path: str,
                       confidence_normalizer: Callable) -> List[str]:
    """
    Generate output texts based on the predictions and save to files.

    Parameters
    ----------
    predictions : List[Tuple[float, str]]
        List of tuples containing confidence and predicted text for each image.
    groups : List[str]
        List of group IDs for each image.
    identifiers : List[str]
        List of identifiers for each image.
    output_path : str
        Base path where prediction outputs should be saved.
    confidence_normalizer : Callable
        Function to normalize the confidence of the predictions.

    Returns
    -------
    List[str]
        List of output texts for each image.

    Side Effects
    ------------
    - Creates directories for groups if they don't exist.
    - Saves output texts to files within the respective group directories.
    - Logs messages regarding directory creation and saving.
    """

    logger = logging.getLogger(__name__)

    outputs = []
    for i, (confidence, pred_text) in enumerate(predictions):
        group_id = groups[i]
        identifier = identifiers[i]
        confidence = confidence_normalizer(confidence, pred_text)

        text = f"{identifier}\t{str(confidence)}\t{pred_text}"
        outputs.append(text)

        # Output the text to a file
        output_dir = os.path.join(output_path, group_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
        with open(os.path.join(output_dir, identifier + ".txt"), "w") as f:
            f.write(text + "\n")

    return outputs
