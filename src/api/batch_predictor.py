# Imports

# > Standard library
import logging
import multiprocessing
import json
import os
import sys
import time
import uuid
import multiprocessing as mp

from typing import List, Tuple

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

# > Local imports
from model.management import load_model_from_directory  # noqa
from setup.environment import initialize_strategy  # noqa


def setup_gpu_environment(gpus: str) -> List[tf.config.PhysicalDevice]:
    """
    Configure the GPU environment for TensorFlow.

    Parameters
    ----------
    gpus : str
        IDs of GPUs to be used (comma-separated). Use '-1' for CPU only or
        'all' for all available GPUs.

    Returns
    -------
    List[tf.config.PhysicalDevice]
        List of active GPUs configured for TensorFlow.
    """
    gpu_devices = tf.config.list_physical_devices("GPU")
    logging.info("Available GPUs: %s", gpu_devices)

    if gpus == "-1":
        active_gpus = []
    elif gpus.lower() == "all":
        active_gpus = gpu_devices
    else:
        gpu_indices = gpus.split(",")
        active_gpus = [
            gpu for i, gpu in enumerate(gpu_devices) if str(i) in gpu_indices
        ]

    if active_gpus:
        logging.info("Using GPU(s): %s", active_gpus)
    else:
        logging.info("Using CPU")

    tf.config.set_visible_devices(active_gpus, "GPU")
    for gpu in active_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    return active_gpus


def get_model_channels(config_path: str) -> int:
    """
    Retrieve the number of input channels for a model from its configuration file.

    Parameters
    ----------
    config_path : str
        Directory path containing the 'config.json' file.

    Returns
    -------
    int
        Number of input channels specified in the configuration.

    Raises
    ------
    FileNotFoundError
        If 'config.json' does not exist in the specified directory.
    ValueError
        If the number of channels is not found in the configuration file.
    """
    config_path = os.path.join(config_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found in the directory: {config_path}"
        )

    # Load the configuration file
    with open(config_path, "r", encoding="UTF-8") as file:
        config = json.load(file)

    # Extract the number of channels
    # First, check the "model_channels" key, then the "args" key
    num_channels = config.get(
        "model_channels", config.get("args", {}).get("channels", None)
    )
    if num_channels is None:
        raise ValueError("Number of channels not found in the config file.")

    logging.debug("Number of channels retrieved: %d", num_channels)
    return num_channels


def update_channels(model_path: str) -> int:
    """
    Update and retrieve the number of channels from the model's configuration.

    Parameters
    ----------
    model_path : str
        Path to the model directory containing 'config.json'.

    Returns
    -------
    int
        Updated number of input channels.

    Raises
    ------
    FileNotFoundError
        If 'config.json' does not exist in the model directory.
    ValueError
        If the number of channels is not specified in the configuration.
    """

    try:
        num_channels = get_model_channels(model_path)
        logging.debug("New number of channels: %s", num_channels)
        return num_channels
    except Exception as e:
        logging.error("Error updating channels: %s", e)
        raise e


def create_model(
    base_model_dir: str, model_path: str, strategy: tf.distribute.Strategy
) -> (tf.keras.Model, int):
    """
    Load a pre-trained TensorFlow model within the given distribution strategy scope.

    Parameters
    ----------
    base_model_dir : str
        Base directory where models are stored.
    model_path : str
        Relative path to the specific model directory within `base_model_dir`.
    strategy : tf.distribute.Strategy
        TensorFlow distribution strategy for model loading.

    Returns
    -------
    tuple
        A tuple containing the loaded `tf.keras.Model` and the number of input channels.

    Raises
    ------
    Exception
        Propagates any exception that occurs during model loading.
    """
    with strategy.scope():
        try:
            model_location = os.path.join(base_model_dir, model_path)
            model = load_model_from_directory(model_location, compile=False)
            num_channels = update_channels(model_location)
            logging.info(
                "Model '%s' loaded successfully with %d channels.",
                model.name,
                num_channels,
            )
        except Exception as e:
            logging.error("Error loading model from '%s': %s", model_path, e)
            raise
    return model, num_channels


def process_sample(
    image_bytes: tf.Tensor,
    group_id: tf.Tensor,
    identifier: tf.Tensor,
    model: tf.Tensor,
    whitelist: tf.Tensor,
    num_channels: int,
) -> tuple:
    """
    Preprocess a single sample for the dataset.

    Parameters
    ----------
    image_bytes : tf.Tensor
        Raw image bytes.
    group_id : tf.Tensor
        Group identifier.
    identifier : tf.Tensor
        Sample identifier.
    model : tf.Tensor
        Model path.
    whitelist : tf.Tensor
        Whitelist metadata.
    num_channels : int
        Number of channels expected in the image.

    Returns
    -------
    tuple
        A tuple containing the processed image and associated metadata.
    """
    try:
        image = tf.io.decode_image(
            image_bytes, channels=num_channels, expand_animations=False
        )
    except tf.errors.InvalidArgumentError:
        image = tf.zeros([64, 64, num_channels], dtype=tf.float32)
        logging.error(
            "Invalid image for identifier: %s", identifier.numpy().decode("utf-8")
        )

    # Resize and normalize the image
    image = tf.image.resize(
        image, [64, tf.constant(99999, dtype=tf.int32)], preserve_aspect_ratio=True
    )
    image = tf.cast(image, tf.float32) / 255.0

    # Resize and pad the image
    image = tf.image.resize_with_pad(
        image, 64, tf.shape(image)[1] + 50, method=tf.image.ResizeMethod.BILINEAR
    )

    # Normalize the image
    image = 0.5 - image

    # Transpose the image dimensions if necessary
    image = tf.transpose(image, perm=[1, 0, 2])  # From HWC to WHC

    return image, group_id, identifier, model, whitelist


def data_generator(
    request_queue: multiprocessing.Queue,
    current_model_path_holder: list,
    stop_event: multiprocessing.Event,
    patience: int,
):
    """
    Generator that yields data from the request queue, ensuring batch consistency
    with the current model.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue containing incoming requests.
    current_model_path_holder : list
        Single-element list holding the current model path for mutability.
    stop_event : multiprocessing.Event
        Event to signal the generator to stop.
    patience : int
        Time in seconds to wait for new requests before yielding the current batch.

    Yields
    ------
    tuple
        A tuple of image_bytes, group_id, identifier, model, and whitelist.
    """
    logging.debug("New data generator started")
    time_since_last_request = time.time()
    has_data = False

    while not stop_event.is_set():
        if not request_queue.empty():
            try:
                time_since_last_request = time.time()
                data = request_queue.get()
                # Unpack to normalize values before yielding
                image_bytes, group_id, identifier, new_model_path, whitelist = data

                # Ensure model path is always a string for tf.data (no None)
                if new_model_path is None:
                    new_model_path = current_model_path_holder[0]

                # Re-pack the possibly updated tuple
                data = (image_bytes, group_id, identifier, new_model_path, whitelist)

                if new_model_path != current_model_path_holder[0]:
                    request_queue.put(data)
                    logging.info(
                        "Model changed to '%s'. Switching generator.", new_model_path
                    )
                    current_model_path_holder[0] = new_model_path
                    break  # Exit to allow dataset recreation with the new model

                has_data = True
                yield data
            except Exception as e:
                logging.error("Error retrieving data from queue: %s", e)
        else:
            time.sleep(0.01)  # Prevent busy waiting

            if time.time() - time_since_last_request > patience and has_data:
                logging.debug(
                    "No new requests for %d seconds. Yielding remaining data.", patience
                )
                break

    logging.debug("Data generator stopped")


def create_dataset(
    request_queue: multiprocessing.Queue,
    batch_size: int,
    current_model_path_holder: list,
    num_channels: int,
    stop_event: multiprocessing.Event,
    patience: int,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from the request queue.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue containing incoming requests.
    batch_size : int
        Number of samples per batch.
    current_model_path_holder : list
        Single-element list holding the current model path.
    num_channels : int
        Number of channels in the input images.
    stop_event : multiprocessing.Event
        Event to signal the dataset creation to stop.
    patience : int
        Time in seconds to wait for new requests before yielding the current batch.

    Returns
    -------
    tf.data.Dataset
        Prepared TensorFlow dataset for batch processing.
    """
    logging.debug("Creating TensorFlow dataset")

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            request_queue, current_model_path_holder, stop_event, patience
        ),
        output_types=(tf.string, tf.string, tf.string, tf.string, tf.string),
        output_shapes=((), (), (), (), (None,)),
    )

    dataset = dataset.map(
        lambda image, group_id, identifier, model, metadata: process_sample(
            image, group_id, identifier, model, metadata, num_channels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            [None, None, num_channels],  # Image shape
            [],  # group_id
            [],  # identifier
            [],  # model
            [None],  # metadata
        ),
        padding_values=(
            tf.constant(-10, dtype=tf.float32),  # Image padding value
            tf.constant("", dtype=tf.string),  # group_id padding
            tf.constant("", dtype=tf.string),  # identifier padding
            tf.constant("", dtype=tf.string),  # model padding
            tf.constant("", dtype=tf.string),  # metadata padding
        ),
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logging.debug("TensorFlow dataset created successfully")

    return dataset


def output_prediction_error(
    output_path: str, group_id: str, identifier: str, text: str
):
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
    with open(
        os.path.join(output_dir, identifier + ".error"), "w", encoding="utf-8"
    ) as f:
        f.write(str(text) + "\n")


def predict(
    model: tf.keras.Model, batch_images: tf.Tensor, batch_id: str
) -> np.ndarray:
    """
    Make predictions on a batch of images using the provided model.

    Parameters
    ----------
    model : TensorFlow model
        The model used for making predictions.
    batch_images : tf.Tensor
        A tensor of images for which predictions need to be made.
    batch_id : str
        Unique identifier for the current batch.

    Returns
    -------
    np.ndarray
        An array of predictions made by the model.
    """

    logging.info("Predicting batch of size %d (%s)", len(batch_images), batch_id)
    t1 = time.time()
    encoded_predictions = model.predict_on_batch(batch_images)

    logging.info(
        "Made %d predictions in %.2f seconds (%s)",
        len(encoded_predictions),
        time.time() - t1,
        batch_id,
    )
    logging.debug("Predictions: %s", encoded_predictions)
    return encoded_predictions


def safe_predict(
    model: tf.keras.Model,
    predicted_queue: multiprocessing.Queue,
    batch_images: tf.Tensor,
    batch_info: List[Tuple[str, str]],
    output_path: str,
    batch_id: str,
) -> List[str]:
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
    batch_id : str
        Unique identifier for the current batch.

    Returns
    -------
    List
        A list of predictions made by the model. If an image causes an OOM
        error, it is skipped, and no prediction is returned for it.
    """

    try:
        return predict(model, batch_images, batch_id)

    except tf.errors.ResourceExhaustedError as e:
        # If the batch size is 1 and still causing OOM, then skip the image and
        # return an empty list
        if len(batch_images) == 1:
            logging.error(
                "OOM error with single image. Skipping image %s.", batch_info[0][1]
            )

            output_prediction_error(output_path, batch_info[0][0], batch_info[0][1], e)
            return []

        logging.warning(
            "OOM error with batch size %s. Splitting batch %s in half and retrying.",
            len(batch_images),
            batch_id,
        )

        # Splitting batch in half
        mid_index = len(batch_images) // 2
        first_half_images = batch_images[:mid_index]
        second_half_images = batch_images[mid_index:]
        first_half_info = batch_info[:mid_index]
        second_half_info = batch_info[mid_index:]

        # Recursive calls for each half
        first_half_predictions = safe_predict(
            model,
            predicted_queue,
            first_half_images,
            first_half_info,
            output_path,
            batch_id,
        )
        second_half_predictions = safe_predict(
            model,
            predicted_queue,
            second_half_images,
            second_half_info,
            output_path,
            batch_id,
        )

        return np.concatenate((first_half_predictions, second_half_predictions))
    except Exception as e:
        logging.error("Error predicting batch %s: %s", batch_id, e)
        for group, identifier in batch_info:
            output_prediction_error(output_path, group, identifier, e)

        return []


def batch_prediction_worker(
    request_queue: multiprocessing.Queue,
    predicted_queue: multiprocessing.Queue,
    base_model_dir: str,
    initial_model_path: str,
    error_output_path: str,
    stop_event: multiprocessing.Event,
    gpus: str = "0",
    batch_size: int = 32,
    patience: int = 1,
):
    """
    Worker process for performing batch predictions on images.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    predicted_queue : multiprocessing.Queue
        Queue to which predictions are put.
    base_model_dir : str
        Base directory where models are stored.
    initial_model_path : str
        Initial model path relative to `base_model_dir`.
    error_output_path : str
        Base path where prediction errors should be saved.
    stop_event : multiprocessing.Event
        Event to signal the worker to stop processing.
    gpus : str, optional
        IDs of GPUs to be used (comma-separated). Use '-1' for CPU only or 'all'
        for all available GPUs. Default is '0'.
    batch_size : int, optional
        Number of samples per batch. Default is 32.
    patience : int, optional
        Time in seconds to wait for new requests before yielding the current batch. Default is 1.

    Side Effects
    ------------
    - Logs various messages regarding the batch processing status.
    - Loads and reloads models as needed.
    - Puts prediction results into `predicted_queue`.
    """
    logging.info("Batch prediction worker started")

    # Configure GPU environment
    active_gpus = setup_gpu_environment(gpus)
    strategy = initialize_strategy(use_float32=False, active_gpus=active_gpus)

    # Load the initial model
    current_model_path_holder = [initial_model_path]
    model, num_channels = create_model(
        base_model_dir, current_model_path_holder[0], strategy
    )

    try:
        while not stop_event.is_set():
            dataset = create_dataset(
                request_queue,
                batch_size,
                current_model_path_holder,
                num_channels,
                stop_event,
                patience,
            )

            for batch in dataset:
                if stop_event.is_set():
                    break

                (
                    images,
                    batch_groups,
                    batch_identifiers,
                    batch_models,
                    batch_whitelist,
                ) = batch
                # Check for model updates
                model_path = batch_models[0].numpy().decode("utf-8")
                if (
                    model_path != current_model_path_holder[0]
                    and model_path is not None
                ):
                    logging.info(
                        "Model switch detected. Replacing old model '%s' with model '%s'.",
                        current_model_path_holder,
                        model_path,
                    )
                    current_model_path_holder[0] = model_path
                    model, num_channels = create_model(
                        base_model_dir, model_path, strategy
                    )
                    logging.debug("Model '%s' loaded successfully.", model_path)

                # Perform predictions
                batch_id = str(uuid.uuid4())
                encoded_predictions = safe_predict(
                    model,
                    predicted_queue,
                    images,
                    list(zip(batch_groups, batch_identifiers)),
                    error_output_path,
                    batch_id,
                )
                logging.debug("Predictions made for batch %s", batch_id)
                predicted_queue.put(
                    (
                        encoded_predictions,
                        batch_groups,
                        batch_identifiers,
                        model_path,
                        batch_id,
                        batch_whitelist,
                    )
                )
    except Exception as e:
        logging.error("Error in batch prediction worker: %s", e)
        raise e
    finally:
        logging.info("Batch prediction worker stopped")
