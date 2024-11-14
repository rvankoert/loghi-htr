# Imports

# > Standard library
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import multiprocessing
from multiprocessing.queues import Empty
import os
import time
import uuid

# > Third-party dependencies
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def image_preparation_worker(batch_size: int,
                             request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             base_model_dir: str,
                             model_name: str,
                             patience: float,
                             stop_event: multiprocessing.Event):
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
    base_model_dir : str
        Path to the base model directory.
    model_name : str
        Path to the initial model used for image preparation.
    patience : float
        Max time to wait for new images.

    Side Effects
    ------------
    - Logs various messages regarding the image preparation status.
    """

    logging.info("Image preparation process started")

    # Disable GPU visibility to prevent memory allocation issues
    tf.config.set_visible_devices([], 'GPU')

    # Define the model path
    model_location = os.path.join(base_model_dir, model_name)

    # Define the number of channels for the images
    num_channels = update_channels(model_location)

    # Initialize the metadata and whitelist as default values
    metadata, whitelist = {}, []

    try:
        while not stop_event.is_set():
            num_channels, model_location, metadata, whitelist = \
                fetch_and_prepare_images(request_queue, prepared_queue,
                                         batch_size, patience, num_channels,
                                         base_model_dir, model_name, metadata,
                                         whitelist, stop_event)

    except Exception as e:
        logging.error("Exception in image preparation worker: %s", e)
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
        logging.debug("New number of channels: %s", num_channels)
        return num_channels
    except Exception as e:
        logging.error("Error retrieving number of channels: %s", e)
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
                        request_queue: multiprocessing.Queue,
                        batch_images: list,
                        batch_groups: list,
                        batch_identifiers: list,
                        batch_metadata: list,
                        old_channels: int,
                        base_model_dir: str,
                        new_model: str,
                        old_model: str) -> (int, str):
    """
    Handles the change of the image processing model.

    Parameters
    ----------
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    batch_images : list
        Current batch of images being processed.
    batch_groups : list
        Group information for each image in the batch.
    batch_identifiers : list
        Identifiers for each image in the batch.
    batch_metadata : list
        Metadata for each image in the batch.
    old_channels : int
        Number of channels for the old model.
    base_model_dir : str
        Path to the base model directory.
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

    new_model_path = os.path.join(base_model_dir, new_model)

    # If there are images in the current batch, process them before changing
    # the model
    if batch_images:
        logging.info(
            "Processing the current batch of %s images before model change.",
            len(batch_images))
        pad_and_queue_batch(old_model, batch_images, batch_groups,
                            batch_identifiers, batch_metadata, old_channels,
                            prepared_queue, request_queue)

    # Update the model channels
    num_channels = update_channels(new_model_path)

    return num_channels, new_model


def fetch_and_prepare_images(request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             batch_size: int,
                             patience: float,
                             num_channels: int,
                             base_model_dir: str,
                             current_model_name: str,
                             metadata: dict,
                             old_whitelist: list,
                             stop_event: multiprocessing.Event) -> (int, str):
    """
    Fetches and prepares images for processing. We pass the current model, the
    current number of channels, the current metadata, and the current whitelist
    to this function so that we can detect changes in these values without
    having to reload the configuration file every time.

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
    metadata : dict
        Metadata for the images.
    old_whitelist : list
        Whitelist for the images.
    stop_event : multiprocessing.Event
        Event to signal the worker process to stop.

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

    while not stop_event.is_set():
        try:
            image, group, identifier, new_model_name, whitelist = \
                request_queue.get(timeout=0.1)
            logging.debug("Retrieved %s from request_queue", identifier)

            # Metadata change detection
            # If the metadata is None or the model has changed or the
            # whitelist has changed, update the metadata
            if (new_model_name != current_model_name and new_model_name is not None) \
                    or whitelist != old_whitelist:
                # Update the metadata
                model_path = os.path.join(base_model_dir, new_model_name)
                logging.info("Detected metadata change. Updating metadata.")
                metadata = fetch_metadata(whitelist, model_path)
                old_whitelist = whitelist
                logging.debug("Metadata updated: %s", metadata)

            # Model change detection
            # If the model has changed, process the current batch before
            # continuing
            if new_model_name and new_model_name != current_model_name:
                num_channels, current_model_name = \
                    handle_model_change(prepared_queue,
                                        request_queue,
                                        batch_images,
                                        batch_groups,
                                        batch_identifiers,
                                        batch_metadata,
                                        num_channels,
                                        base_model_dir,
                                        new_model_name,
                                        current_model_name)

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
                              "%s s", time.time() - last_image_time)

            # Check if there's at least one image and max wait time is exceeded
            if last_image_time is not None and \
                    (time.time() - last_image_time) >= patience:
                break
    else:
        # If the stop event is set, break the loop
        return num_channels, current_model_name, metadata, old_whitelist

    # Pad and queue the batch
    pad_and_queue_batch(current_model_name, batch_images, batch_groups,
                        batch_identifiers, batch_metadata, num_channels,
                        prepared_queue, request_queue)

    return num_channels, current_model_name, metadata, old_whitelist


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
    with open(config_path, 'r', encoding="utf-8") as file:
        config = json.load(file)

    # Initialize a dictionary to store the found values
    values = {}

    # Function to recursively search for keys in a nested dictionary
    def search_key(data, key):
        if key in data:
            return data[key]
        for sub_value in data.values():
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
                logging.warning("Key %s not found in config file. "
                                "Recording 'NOT_FOUND'", key)
                values[key] = "NOT_FOUND"
            else:
                # Otherwise, record the found value
                values[key] = value

    return values


def prepare_image(image_bytes: bytes,
                  num_channels: int) -> np.ndarray:
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
        3 for RGB, 4 for RGBA).

    Returns
    -------
    np.ndarray
        Prepared image array.
    """

    # Load the image using TensorFlow
    image = tf.image.decode_image(image_bytes, channels=num_channels,
                                  expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Convert TensorFlow tensor to PIL Image
    if num_channels == 1:
        image = tf.squeeze(image, axis=-1)  # Remove the channel dimension
        image = Image.fromarray(
            (image.numpy() * 255).astype(np.uint8), mode='L')
    else:
        image = Image.fromarray((image.numpy() * 255).astype(np.uint8))

    # Convert to desired number of channels
    if num_channels == 1:
        image = image.convert("L")
    elif num_channels == 3:
        image = image.convert("RGB")
    elif num_channels == 4:
        image = image.convert("RGBA")

    # Resize while preserving aspect ratio
    target_height = 64
    aspect_ratio = image.width / image.height
    target_width = int(target_height * aspect_ratio)
    image = image.resize((target_width, target_height), Image.BILINEAR)

    # Calculate padding sizes
    padding_height = target_height
    padding_width = target_width + 50

    # Create new image with padding, centering the original image
    padded_image = ImageOps.pad(
        image, (padding_width, padding_height), color=0, centering=(0.5, 0.5))

    # Normalize the image
    image_array = np.array(padded_image) / 255.0
    image_array = 0.5 - image_array

    # Ensure the image has the right number of channels
    if num_channels == 1:
        image_array = np.expand_dims(image_array, axis=-1)

    # Transpose the image
    image_array = np.transpose(image_array, (1, 0, 2))

    return image_array


def pad_images(images: list, num_channels: int) -> np.ndarray:
    """
    Pad a list of images to have the same dimensions.

    Parameters
    ----------
    images : list
        List of image arrays to be padded.
    num_channels : int
        Number of channels in the images.

    Returns
    -------
    np.ndarray
        Padded batch of images.
    """
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)

    padded_batch = np.full((len(images), max_height, max_width, num_channels),
                           -10.0, dtype=np.float32)

    for i, image in enumerate(images):
        padded_batch[i, :image.shape[0], :image.shape[1], ...] = image

    return padded_batch


def pad_and_queue_batch(model_path: str,
                        batch_images: list,
                        batch_groups: list,
                        batch_identifiers: list,
                        batch_metadata: list,
                        channels: int,
                        prepared_queue: multiprocessing.Queue,
                        request_queue: multiprocessing.Queue) -> None:
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
    channels : int
        Number of channels for the images.
    prepared_queue : multiprocessing.Queue
        Queue to which the padded batch should be pushed.
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    """

    # Generate a unique identifier for the batch
    batch_id = str(uuid.uuid4())

    # Prepare images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        prepared_images = list(executor.map(
            lambda image: prepare_image(image, channels), batch_images))

    # Pad the batch
    padded_images = pad_images(prepared_images, channels)

    # Push the prepared batch to the prepared_queue
    prepared_queue.put((padded_images, batch_groups, batch_identifiers,
                        batch_metadata, model_path, batch_id))
    logging.info("Prepared batch %s (%s items) for "
                 "prediction", batch_id, len(batch_images))
    logging.info("%s items waiting to be processed", request_queue.qsize())
