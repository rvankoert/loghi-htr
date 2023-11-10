# Imports

# > Standard library
import json
import logging
import multiprocessing
from multiprocessing.queues import Empty
import os

# > Third-party dependencies
import numpy as np
import tensorflow as tf


def image_preparation_worker(batch_size: int,
                             request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue):
    """
    Worker process to prepare images for batch processing.

    Continuously fetches raw images from the request queue, processes them, and
    pushes the prepared images to the prepared queue.

    Parameters
    ----------
    batch_size : int
        Number of images to process in a batch.
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.

    Side Effects
    ------------
    - Logs various messages regarding the image preparation status.
    """

    logger = logging.getLogger(__name__)
    logger.info("Image Preparation Worker process started")

    # Disable GPU visibility to prevent memory allocation issues
    tf.config.set_visible_devices([], 'GPU')

    # Define the maximum time to wait for new images
    TIMEOUT_DURATION = 1
    MAX_WAIT_COUNT = 1

    wait_count = 0
    old_model = None
    batch_images, batch_groups, batch_identifiers = [], [], []

    try:
        while True:
            while len(batch_images) < batch_size:
                try:
                    image, group, identifier, model_path = request_queue.get(
                        timeout=TIMEOUT_DURATION)
                    logger.debug(f"Retrieved {identifier} from request_queue")

                    # Check if the model has changed
                    if model_path != old_model:
                        logger.info(
                            "Model changed, adjusting image preparation")
                        if batch_images:
                            # Add the existing batch to the prepared_queue
                            logger.info(
                                f"Sending old batch of {len(batch_images)} "
                                "images")

                            # Reset the batches after sending
                            batch_images, batch_groups, batch_identifiers = \
                                pad_and_queue_batch(batch_images, batch_groups,
                                                    batch_identifiers,
                                                    prepared_queue, old_model)

                        old_model = model_path
                        num_channels = update_channels(model_path, logger)

                    image = prepare_image(image, num_channels)
                    logger.debug(
                        f"Prepared image {identifier} with shape: "
                        f"{image.shape}")

                    # Append the image to the batch
                    batch_images.append(image)
                    batch_groups.append(group)
                    batch_identifiers.append(identifier)

                    request_queue.task_done()
                    wait_count = 0

                # If no new images are available, wait for a while
                except Empty:
                    wait_count += 1
                    logger.debug(
                        "Time without new images: "
                        f"{wait_count * TIMEOUT_DURATION} seconds")

                    if wait_count > MAX_WAIT_COUNT and len(batch_images) > 0:
                        break

            # Add the existing batch to the prepared_queue
            batch_images, batch_groups, batch_identifiers = \
                pad_and_queue_batch(batch_images, batch_groups,
                                    batch_identifiers, prepared_queue,
                                    old_model)
            logger.debug(
                f"{request_queue.qsize()} images waiting to be processed")

    except KeyboardInterrupt:
        logger.warning(
            "Image Preparation Worker process interrupted. Exiting...")
    except Exception as e:
        logger.error(f"Error: {e}")


def pad_and_queue_batch(batch_images: np.ndarray,
                        batch_groups: list,
                        batch_identifiers: list,
                        prepared_queue: multiprocessing.Queue,
                        model_path: str) -> tuple:
    """
    Pad and queue a batch of images for prediction.

    Parameters
    ----------
    batch_images : np.ndarray
        Batch of images to be padded and queued.
    batch_groups : list
        List of groups to which the images belong.
    batch_identifiers : list
        List of identifiers for the images.
    prepared_queue : multiprocessing.Queue
        Queue to which the padded batch should be pushed.
    model_path : str
        Path to the model used for image preparation.

    Returns
    -------
    tuple
        Tuple containing the empty batch images, groups, and identifiers.
    """

    logger = logging.getLogger(__name__)

    # Pad the batch
    padded_batch = pad_batch(batch_images)

    # Push the prepared batch to the prepared_queue
    prepared_queue.put(
        (np.array(padded_batch), batch_groups, batch_identifiers, model_path))
    logger.debug("Pushed prepared batch to prepared_queue")
    logger.debug(
        f"{prepared_queue.qsize()} batches ready for prediction")

    return [], [], []


def pad_batch(batch_images: np.ndarray) -> np.ndarray:
    """
    Pad a batch of images to the same width.

    Parameters
    ----------
    batch_images : np.ndarray
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
        batch_images[i] = pad_to_width(
            batch_images[i], max_width, -10)

    return batch_images


def update_channels(model_path: str, logger):
    """
    Update the model used for image preparation.

    Parameters
    ----------
    model_path : str
        The path to the directory containing the 'config.json' file.
        The function will append "/config.json" to this path.
    logger : logging.Logger
        Logger object to log messages.
    """

    try:
        num_channels = get_model_channels(model_path)
        logger.debug(
            f"New number of channels: "
            f"{num_channels}")
        return num_channels
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(
            "Error retrieving number of channels. "
            "Exiting...")
        return


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


def prepare_image(image_bytes: bytes,
                  num_channels: int) -> tf.Tensor:
    """
    Prepare a raw image for batch processing.

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

    image = tf.io.decode_jpeg(image_bytes, channels=num_channels)

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
    num_channels = config.get("args", {}).get("channels")
    if num_channels is None:
        raise ValueError("Number of channels not found in the config file.")

    return num_channels
