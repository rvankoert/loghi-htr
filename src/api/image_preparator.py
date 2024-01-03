# Imports

# > Standard library
import json
import logging
import multiprocessing
from multiprocessing.queues import Empty
import os
import time
import uuid

# > Third-party dependencies
import numpy as np
import tensorflow as tf


def image_preparation_worker(batch_size: int,
                             request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             model_path: str,
                             patience: float):
    """
    Worker process to prepare images for batch processing.

    Continuously fetches raw images from the request queue, processes them, and
    pushes the prepared images to the prepared queue.

    Parameters
    ----------
    batch_size : int
        Max number of images to process in a batch.
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.
    model_path : str
        Path to the initial model used for image preparation.
    patience : float
        Max time to wait for new images.

    Side Effects
    ------------
    - Logs various messages regarding the image preparation status.
    """

    logging.info("Image Preparation Worker process started")

    # Disable GPU visibility to prevent memory allocation issues
    tf.config.set_visible_devices([], 'GPU')

    # Define the number of channels for the images
    num_channels = update_channels(model_path)

    # Define the model path
    model = model_path

    # Initialize the metadata and whitelist as None
    metadata, whitelist = None, None

    try:
        while True:
            num_channels, model, metadata, whitelist = \
                fetch_and_prepare_images(request_queue, prepared_queue,
                                         batch_size, patience, num_channels,
                                         model, metadata, whitelist)

    except Exception as e:
        logging.error(f"Exception in image preparation worker: {e}")
        raise


def update_channels(model_path: str) -> int:
    """
    Update the model used for image preparation.

    Parameters
    ----------
    model_path : str
        The path to the directory containing the 'config.json' file.
        The function will append "/config.json" to this path.
    """

    try:
        num_channels = get_model_channels(model_path)
        logging.debug(f"New number of channels: {num_channels}")
        return num_channels
    except Exception as e:
        logging.error(f"Error retrieving number of channels: {e}")
        raise e


def get_model_channels(config_path: str) -> int:
    """
    Retrieve the number of input channels for a model from a configuration
    file.

    This function reads a JSON configuration file located in the specified
    directory to extract the number of input channels used by the model.

    Parameters
    ----------
    config_path : str
        The path to the directory containing the 'config.json' file.
        The function will append "/config.json" to this path.

    Returns
    -------
    int
        The number of input channels specified in the configuration file.

    Raises
    ------
    FileNotFoundError
        If the 'config.json' file does not exist in the given directory.

    ValueError
        If the number of channels is not found or not specified in the
        'config.json' file.
    """

    config_path = os.path.join(config_path, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found in the directory: {config_path}")

    # Load the configuration file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Extract the number of channels
    # First, check the "model_channels" key, then the "args" key
    num_channels = config.get("model_channels",
                              config.get("args", {}).get("channels", None))
    if num_channels is None:
        raise ValueError(
            "Number of channels not found in the config file.")

    return num_channels


def handle_model_change(prepared_queue: multiprocessing.Queue,
                        batch_images: list,
                        batch_groups: list,
                        batch_identifiers: list,
                        batch_metadata: list,
                        new_model: str,
                        old_model: str) -> (int, str):
    """
    Handles the change of the image processing model.

    Parameters
    ----------
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.
    batch_images : list
        Current batch of images being processed.
    batch_groups : list
        Group information for each image in the batch.
    batch_identifiers : list
        Identifiers for each image in the batch.
    batch_metadata : list
        Metadata for each image in the batch.
    new_model : str
        Path of the new model.
    old_model : str
        Path of the old model.

    Returns
    -------
    int
        Number of channels for the new model.
    str
        Path of the new model.
    """

    logging.info("Detected model change. Switching to the new model.")

    # If there are images in the current batch, process them before changing
    # the model
    if batch_images:
        logging.info(
            f"Processing the current batch of {len(batch_images)} images "
            "before model change.")
        pad_and_queue_batch(old_model, batch_images, batch_groups,
                            batch_identifiers, batch_metadata, prepared_queue)

        # Clearing the current batch
        batch_images.clear()
        batch_groups.clear()
        batch_identifiers.clear()
        batch_metadata.clear()

    # Update the model channels
    num_channels = update_channels(new_model)

    return num_channels, new_model


def fetch_and_prepare_images(request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             batch_size: int,
                             patience: float,
                             num_channels: int,
                             current_model: str,
                             metadata,
                             old_whitelist) -> (int, str):
    """
    Fetches and prepares images for processing.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.
    batch_size : int
        Max number of images to process in a batch.
    patience : float
        Max time to wait for new images.
    num_channels : int
        Number of channels for the images.
    current_model : str
        Path of the old model.

    Returns
    -------
    int
        Number of channels for the current model.
    str
        Path of the current model.
    """

    last_image_time = None
    batch_images, batch_groups, batch_identifiers, batch_metadata \
        = [], [], [], []

    while True:
        try:
            image, group, identifier, new_model, whitelist = \
                request_queue.get(timeout=0.1)
            logging.debug(f"Retrieved {identifier} from request_queue")

            # Metadata change detection
            # If the metadata is None or the model has changed or the
            # whitelist has changed, update the metadata
            if metadata is None or new_model != current_model or \
                    whitelist != old_whitelist:
                # Update the metadata
                logging.info("Detected metadata change. Updating metadata.")
                metadata = fetch_metadata(whitelist, new_model)
                old_whitelist = whitelist
                logging.debug(f"Metadata updated: {metadata}")

            # Model change detection
            # If the model has changed, process the current batch before
            # continuing
            if new_model and new_model != current_model:
                num_channels, current_model = \
                    handle_model_change(prepared_queue,
                                        batch_images,
                                        batch_groups,
                                        batch_identifiers,
                                        batch_metadata,
                                        new_model,
                                        current_model)

            # Prepare the image
            image = prepare_image(image, num_channels)
            batch_images.append(image)
            batch_groups.append(group)
            batch_identifiers.append(identifier)
            batch_metadata.append(metadata)

            # Reset the last image time
            last_image_time = time.time()

            # Check if batch is full or max wait time is exceeded or model
            # change is detected
            if len(batch_images) >= batch_size or \
                    (time.time() - last_image_time) >= patience:
                break

        except Empty:
            # If there are no images in the queue, log the time
            if last_image_time is not None:
                logging.debug("Time without new images: "
                              f"{time.time() - last_image_time}s")

            # Check if there's at least one image and max wait time is exceeded
            if last_image_time is not None and \
                    (time.time() - last_image_time) >= patience:
                break

    # Pad and queue the batch
    pad_and_queue_batch(current_model, batch_images, batch_groups,
                        batch_identifiers, batch_metadata, prepared_queue)

    return num_channels, current_model, metadata, old_whitelist


def prepare_image(image_bytes: bytes,
                  num_channels: int) -> tf.Tensor:
    """
    Prepare a single raw image for batch processing.

    Decodes, resizes, normalizes, pads, and transposes the image for further
    processing.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the image.
    num_channels : int
        Number of channels desired for the output image (e.g., 1 for grayscale,
        3 for RGB).

    Returns
    -------
    tf.Tensor
        Prepared image tensor.
    """

    image = tf.io.decode_image(image_bytes, channels=num_channels)

    # Resize while preserving aspect ratio
    target_height = 64
    aspect_ratio = tf.shape(image)[1] / tf.shape(image)[0]
    target_width = tf.cast(target_height * aspect_ratio, tf.int32)
    image = tf.image.resize(image,
                            [target_height, target_width])

    image = tf.image.resize_with_pad(image,
                                     target_height,
                                     target_width + 50)

    # Normalize the image and something else
    image = 0.5 - (image / 255)

    # Transpose the image
    image = tf.transpose(image, perm=[1, 0, 2])

    return image


def fetch_metadata(whitelist: list, model_path: str):
    """
    Fetch metadata values based on the whitelist keys from a JSON configuration
    file. If a key is not found, 'NOT_FOUND' is recorded as its value.

    Parameters
    ----------
    whitelist : list
        A list of keys to search for in the JSON configuration.
    model_path : str
        Path to the directory containing the JSON configuration file.

    Returns
    -------
    dict
        A dictionary with keys from the whitelist and their corresponding
        values or 'NOT_FOUND'.
    """

    # Define the path to the config.json file
    config_path = os.path.join(model_path, "config.json")

    # Load the configuration file
    with open(config_path, 'r') as file:
        config = json.load(file)

    # Initialize a dictionary to store the found values
    values = {}

    # Function to recursively search for keys in a nested dictionary
    def search_key(data, key):
        if key in data:
            return data[key]
        for sub_key, sub_value in data.items():
            if isinstance(sub_value, dict):
                result = search_key(sub_value, key)
                if result is not None:
                    return result
        return None

    # Check top level and under 'args'
    for key in whitelist:
        if key in config:
            values[key] = config[key]
        elif "args" in config and key in config["args"]:
            values[key] = config["args"][key]
        else:
            # Search in nested structures under 'args'
            value = search_key(config.get("args", {}), key)

            if value is None:
                # If the key is not found, record 'NOT_FOUND'
                logging.warning(f"Key {key} not found in config file. "
                                "Recording 'NOT_FOUND'")
                values[key] = "NOT_FOUND"
            else:
                # Otherwise, record the found value
                values[key] = value

    return values


def pad_and_queue_batch(model_path: str,
                        batch_images: list,
                        batch_groups: list,
                        batch_identifiers: list,
                        batch_metadata: list,
                        prepared_queue: multiprocessing.Queue) -> None:
    """
    Pad and queue a batch of images for prediction.

    Parameters
    ----------
    model_path : str
        Path to the model used for image preparation.
    batch_images : list
        Batch of images to be padded and queued.
    batch_groups : list
        List of groups to which the images belong.
    batch_identifiers : list
        List of identifiers for the images.
    batch_metadata : list
        List of metadata for the images.
    prepared_queue : multiprocessing.Queue
        Queue to which the padded batch should be pushed.
    """

    # Generate a unique identifier for the batch
    batch_id = str(uuid.uuid4())

    # Pad the batch
    padded_batch = pad_batch(batch_images)

    # Push the prepared batch to the prepared_queue
    prepared_queue.put((padded_batch, batch_groups, batch_identifiers,
                        batch_metadata, model_path, batch_id))
    logging.info(f"Prepared batch {batch_id} ({len(batch_images)} items) for "
                 "prediction")
    logging.debug(f"{prepared_queue.qsize()} batches ready for prediction")


def pad_batch(batch_images: list) -> np.ndarray:
    """
    Pad a batch of images to the same width.

    Parameters
    ----------
    batch_images : list
        Batch of images to be padded.

    Returns
    -------
    np.ndarray
        Batch of padded images.
    """

    # Determine the maximum width among all images in the batch
    max_width = max(image.shape[0] for image in batch_images)

    # Resize each image in the batch to the maximum width
    for i in range(len(batch_images)):
        batch_images[i] = pad_to_width(batch_images[i], max_width, -10)

    batch_images = tf.convert_to_tensor(batch_images)

    return batch_images


def pad_to_width(image: tf.Tensor, target_width: int, pad_value: float):
    """
    Pads a transposed image (where the first dimension is width) to a specified
    target width, adding padding equally on the top and bottom sides of the
    image. The padding is applied such that the image content is centered.

    Parameters
    ----------
    image : tf.Tensor
        A 3D TensorFlow tensor representing an image, where the image is
        already transposed such that the width is the first dimension and the
        height is the second dimension.
        The shape of the tensor is expected to be [width, height, channels].
    target_width : int
        The target width to which the image should be padded. If the current
        width of the image is greater than this value, no padding will be
        added.
    pad_value : float
        The scalar value to be used for padding.

    Returns
    -------
    tf.Tensor
        A 3D TensorFlow tensor of the same type as the input 'image', with
        padding applied to reach the target width. The shape of the output
        tensor will be [target_width, original_height, channels].
    """
    current_width = tf.shape(image)[0]

    # Calculate the padding size
    pad_width = target_width - current_width

    # Ensure no negative padding
    pad_width = max(pad_width, 0)

    # Configure padding to add only on the right side
    padding = [[0, pad_width], [0, 0], [0, 0]]

    # Pad the image
    return tf.pad(image, padding, "CONSTANT", constant_values=pad_value)
