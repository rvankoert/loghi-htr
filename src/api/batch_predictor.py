# Imports

# > Standard library
import logging
import multiprocessing
import os
import sys
from typing import Callable, List, Tuple
import gc

# > Third-party dependencies
import tensorflow as tf
from tensorflow.keras import mixed_precision


def batch_prediction_worker(prepared_queue: multiprocessing.JoinableQueue,
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
    prepared_queue : multiprocessing.JoinableQueue
        Queue from which preprocessed images are fetched.
    output_path : str
        Path where predictions should be saved.
    model_path : str
        Path to the initial model file.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Default is '0'.

    Side Effects
    ------------
    - Modifies CUDA_VISIBLE_DEVICES environment variable to control GPU
    visibility.
    - Alters the system path to enable certain imports.
    - Logs various messages regarding the batch processing status.
    """

    # Add parent directory to path for imports
    current_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.dirname(current_path)
    sys.path.append(parent_path)

    from utils.utils import decode_batch_predictions, normalize_confidence

    logger = logging.getLogger(__name__)
    logger.info("Batch Prediction Worker process started")

    # If all GPUs support mixed precision, enable it
    gpus_support_mixed_precision = setup_gpu_environment(gpus, logger)
    if gpus_support_mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        logger.debug("Mixed precision set to 'mixed_float16'")
    else:
        logger.debug(
            "Not all GPUs support efficient mixed precision. Running in "
            "standard mode.")

    strategy = tf.distribute.MirroredStrategy()

    # Create the model and utilities
    try:
        with strategy.scope():
            model, utils = create_model(model_path)
        logger.info("Model created and utilities initialized")
    except Exception as e:
        logger.error(e)
        logger.error("Error creating model. Exiting...")
        return

    total_predictions = 0
    old_model_path = model_path

    try:
        while True:
            batch_images, batch_groups, batch_identifiers, model_path = \
                prepared_queue.get()
            logger.debug(f"Retrieved batch of size {len(batch_images)} from "
                         "prepared_queue")

            batch_info = list(zip(batch_groups, batch_identifiers))

            if model_path != old_model_path:
                old_model_path = model_path
                try:
                    logger.warning("Model changed, adjusting batch prediction")
                    with strategy.scope():
                        model, utils = create_model(model_path)
                    logger.info("Model created and utilities initialized")
                except Exception as e:
                    logger.error(e)
                    logger.error("Error creating model. Exiting...")
                    return

            # Here, make the batch prediction
            try:
                predictions = safe_batch_predict(
                    model, batch_images, batch_info, utils,
                    decode_batch_predictions, output_path,
                    normalize_confidence)
            except Exception as e:
                logger.error(e)
                logger.error("Error making predictions. Skipping batch.")
                logger.error("Failed batch:")
                for group, id in batch_info:
                    logger.error(id)
                    output_prediction_error(output_path, group, id, e)
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
                f"{prepared_queue.qsize()} batches waiting on prediction")

            # Clear the batch images to free up memory
            logger.debug("Clearing batch images and predictions")
            del batch_images
            del predictions
            gc.collect()

    except KeyboardInterrupt:
        logger.warning(
            "Batch Prediction Worker process interrupted. Exiting...")


def setup_gpu_environment(gpus: str, logger: logging.Logger):
    """
    Setup the GPU environment for batch prediction.

    Parameters
    ----------
    gpus : str
        IDs of GPUs to be used (comma-separated).
    logger : logging.Logger
        A logging.Logger object for logging messages.

    Returns
    -------
    bool
        True if all GPUs support mixed precision, False otherwise.
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        logger.warning("No GPUs found. Running in CPU mode.")
        return False
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        if tf.config.experimental.\
                get_device_details(device)['compute_capability'][0] < 7:
            return False
    return True


def create_model(model_path: str) -> Tuple[tf.keras.Model, object]:
    """
    Load a pre-trained model and create utility methods.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file.

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

    from model.custom_layers import ResidualBlock
    from model.model import CERMetric, WERMetric, CTCLoss
    from model.optimization import LoghiLearningRateSchedule
    from utils.utils import Utils, load_model_from_directory

    logger = logging.getLogger(__name__)

    logger.info("Loading model...")
    custom_objects = {
        'CERMetric': CERMetric,
        'WERMetric': WERMetric,
        'CTCLoss': CTCLoss,
        'ResidualBlock': ResidualBlock,
        'LoghiLearningRateSchedule': LoghiLearningRateSchedule
    }
    model = load_model_from_directory(model_path, custom_objects)
    logger.info(f"Model {model.name} loaded successfully")

    if logger.isEnabledFor(logging.DEBUG):
        model.summary()

    try:
        with open(f"{model_path}/charlist.txt") as file:
            charlist = list(char for char in file.read())
    except FileNotFoundError:
        logger.error(f"charlist.txt not found at {model_path}. Exiting...")
        sys.exit(1)

    utils = Utils(charlist, use_mask=True)
    logger.debug("Utilities initialized")

    return model, utils


def safe_batch_predict(model: tf.keras.Model,
                       batch_images: List[tf.Tensor],
                       batch_info: List[Tuple[str, str]],
                       utils: object,
                       decode_batch_predictions: Callable,
                       output_path: str,
                       normalize_confidence: Callable) -> List[str]:
    """
    Attempt to predict on a batch of images using the provided model. If a
    TensorFlow Out of Memory (OOM) error occurs, the batch is split in half and
    each half is attempted again, recursively. If an OOM error occurs with a
    batch of size 1, the offending image is logged and skipped.

    Parameters
    ----------
    model : TensorFlow model
        The model used for making predictions.
    batch_images : List or ndarray
        A list or numpy array of images for which predictions need to be made.
    batch_info : List of tuples
        A list of tuples containing additional information (e.g., group and
        identifier) for each image in `batch_images`.
    utils : module or object
        Utility module/object containing necessary utility functions or
        settings.
    decode_batch_predictions : function
        A function to decode the predictions made by the model.
    output_path : str
        Path where any output files should be saved.
    normalize_confidence : function
        A function to normalize the confidence of the predictions.
    logger : Logger
        A logging.Logger object for logging messages.

    Returns
    -------
    List
        A list of predictions made by the model. If an image causes an OOM
        error, it is skipped, and no prediction is returned for it.
    """

    logger = logging.getLogger(__name__)
    try:
        return batch_predict(
            model, batch_images, batch_info, utils,
            decode_batch_predictions, output_path,
            normalize_confidence)
    except tf.errors.ResourceExhaustedError as e:
        # If the batch size is 1 and still causing OOM, then skip the image and
        # return an empty list
        if len(batch_images) == 1:
            logger.error(
                "OOM error with single image. Skipping image"
                f"{batch_info[0][1]}.")

            output_prediction_error(
                output_path, batch_info[0][0], batch_info[0][1], e)
            return []

        logger.warning(
            f"OOM error with batch size {len(batch_images)}. Splitting batch "
            "in half and retrying.")

        # Splitting batch in half
        mid_index = len(batch_images) // 2
        first_half_images = batch_images[:mid_index]
        second_half_images = batch_images[mid_index:]
        first_half_info = batch_info[:mid_index]
        second_half_info = batch_info[mid_index:]

        # Recursive calls for each half
        first_half_predictions = safe_batch_predict(
            model, first_half_images, first_half_info, utils,
            decode_batch_predictions, output_path,
            normalize_confidence)
        second_half_predictions = safe_batch_predict(
            model, second_half_images, second_half_info, utils,
            decode_batch_predictions, output_path,
            normalize_confidence)

        return first_half_predictions + second_half_predictions


def batch_predict(model: tf.keras.Model,
                  images: List[Tuple[tf.Tensor, str, str]],
                  batch_info: List[Tuple[str, str]],
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

    logger.debug(f"Initial batch size: {len(images)}")

    # Unpack the batch
    groups, identifiers = zip(*batch_info)

    logger.info(f"Making {len(images)} predictions...")
    encoded_predictions = model.predict_on_batch(images)
    logger.debug("Predictions made")

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
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        with open(os.path.join(output_dir, identifier + ".txt"), "w") as f:
            f.write(text + "\n")

    return outputs


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
    with open(os.path.join(output_dir, identifier + ".error"), "w") as f:
        f.write(str(text) + "\n")
