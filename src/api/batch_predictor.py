# Imports

# > Standard library
import logging
import multiprocessing
import os
import sys
from typing import Any, List, Tuple

# > Third-party dependencies
import tensorflow as tf

# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from utils.utils import Utils, decode_batch_predictions, \
        normalize_confidence, load_model_from_directory  # noqa: E402


def create_model(model_path: str, strategy: tf.distribute.Strategy) \
        -> Tuple[tf.keras.Model, object]:
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

    logging.info("Loading model...")
    with strategy.scope():
        try:
            model = load_model_from_directory(model_path, compile=False)
            logging.info(f"Model {model.name} loaded successfully")

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                model.summary()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            sys.exit(1)

    # Load the character list
    charlist_path = f"{model_path}/charlist.txt"
    try:
        with open(charlist_path) as file:
            charlist = [char for char in file.read()]
        utils = Utils(charlist, use_mask=True)
        logging.debug("Utilities initialized")
    except FileNotFoundError:
        logging.error(f"charlist.txt not found at {model_path}. Exiting...")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading utilities: {e}")
        sys.exit(1)

    return model, utils


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
    logging.info(f"Available GPUs: {gpu_devices}")

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
        logging.info(f"Using GPU(s): {active_gpus}")
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, 'GPU')

    # Check if all GPUs support mixed precision
    gpus_support_mixed_precision = True if active_gpus else False
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

    logging.info("Batch Prediction Worker process started")

    # If all GPUs support mixed precision, enable it
    mixed_precision_enabled = setup_gpu_environment(gpus)
    strategy = tf.distribute.MirroredStrategy()

    # Create the model and utilities
    model, utils = create_model(model_path, strategy)

    total_predictions = 0
    old_model_path = model_path

    try:
        while True:
            batch_data = prepared_queue.get()
            logging.debug("Received batch from prepared_queue")

            if model_path != old_model_path:
                old_model_path = model_path
                model, utils = create_model(model_path, strategy)
                logging.info("Model reloaded due to change in model path")

            # Make predictions on the batch
            num_predictions = handle_batch_prediction(
                model, batch_data, utils, output_path)
            total_predictions += num_predictions

            logging.info(f"Made {num_predictions} predictions")
            logging.info(f"Total predictions: {total_predictions}")
            logging.info(f"{prepared_queue.qsize()} batches waiting on "
                         "prediction")

    except KeyboardInterrupt:
        logging.warning(
            "Batch Prediction Worker process interrupted. Exiting...")


def handle_batch_prediction(model: tf.keras.Model,
                            batch_data: Tuple[List[Any], ...],
                            utils: Utils,
                            output_path: str) -> int:
    """
    Handle the batch prediction process.

    Parameters:
    -----------
    model : Any
        The loaded model for predictions.
    batch_data : Tuple[List[Any], ...]
        Tuple containing batch images, groups, and identifiers.
    utils : Any
        Utilities for processing predictions.
    output_path : str
        Path where predictions should be saved.

    Returns:
    --------
    int
        Number of predictions made.
    """

    batch_images, batch_groups, batch_identifiers, _ = batch_data
    batch_info = list(zip(batch_groups, batch_identifiers))

    try:
        predictions = safe_batch_predict(model, batch_images, batch_info,
                                         utils, output_path)
        for prediction in predictions:
            logging.debug(f"Prediction: {prediction}")
        return len(predictions)

    except Exception as e:
        logging.error(e)
        failed_ids = [id for _, id in batch_info]
        logging.error("Error making predictions. Skipping batch:\n" +
                      "\n".join(failed_ids))

        for group, id in batch_info:
            output_prediction_error(output_path, group, id, e)
        return 0


def safe_batch_predict(model: tf.keras.Model,
                       batch_images: List[tf.Tensor],
                       batch_info: List[Tuple[str, str]],
                       utils: Utils,
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
    batch_images : List[tf.Tensor]
        A list or numpy array of images for which predictions need to be made.
    batch_info : List of tuples
        A list of tuples containing additional information (e.g., group and
        identifier) for each image in `batch_images`.
    utils : Utils
        Utility methods for handling predictions.
    output_path : str
        Path where any output files should be saved.

    Returns
    -------
    List
        A list of predictions made by the model. If an image causes an OOM
        error, it is skipped, and no prediction is returned for it.
    """

    try:
        return batch_predict(model, batch_images, batch_info, utils,
                             output_path)
    except tf.errors.ResourceExhaustedError as e:
        # If the batch size is 1 and still causing OOM, then skip the image and
        # return an empty list
        if len(batch_images) == 1:
            logging.error(
                "OOM error with single image. Skipping image"
                f"{batch_info[0][1]}.")

            output_prediction_error(
                output_path, batch_info[0][0], batch_info[0][1], e)
            return []

        logging.warning(
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
    except Exception as e:
        raise e


def batch_predict(model: tf.keras.Model,
                  images: List[tf.Tensor],
                  batch_info: List[Tuple[str, str]],
                  utils: Utils,
                  output_path: str) -> List[str]:
    """
    Process a batch of images using the provided model and decode the
    predictions.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-trained model for predictions.
    images : List[tf.Tensor]
        List of images for which predictions need to be made.
    batch_info : List[Tuple[str, str]]
        List of tuples containing group and identifier for each image in the
        batch.
    utils : Utils
        Utility methods for handling predictions.
    output_path : str
        Path where predictions should be saved.

    Returns
    -------
    List[str]
        List of predicted texts for each image in the batch.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing and prediction
    status.
    """

    logging.debug(f"Initial batch size: {len(images)}")

    # Unpack the batch
    groups, identifiers = zip(*batch_info)

    logging.info(f"Making {len(images)} predictions...")
    encoded_predictions = model.predict_on_batch(images)
    logging.debug("Predictions made")

    logging.debug("Decoding predictions...")
    decoded_predictions = decode_batch_predictions(
        encoded_predictions, utils)[0]
    logging.debug("Predictions decoded")

    logging.debug("Outputting predictions...")
    predicted_texts = output_predictions(decoded_predictions,
                                         groups,
                                         identifiers,
                                         output_path)
    logging.debug("Predictions outputted")

    return predicted_texts


def output_predictions(predictions: List[Tuple[float, str]],
                       groups: List[str],
                       identifiers: List[str],
                       output_path: str) -> List[str]:
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

    outputs = []
    for i, (confidence, pred_text) in enumerate(predictions):
        group_id = groups[i]
        identifier = identifiers[i]
        confidence = normalize_confidence(confidence, pred_text)

        text = f"{identifier}\t{str(confidence)}\t{pred_text}"
        outputs.append(text)

        # Output the text to a file
        output_dir = os.path.join(output_path, group_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Created output directory: {output_dir}")
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
