# Imports

# > Standard library
import logging
import multiprocessing
from multiprocessing.queues import Empty

# > Third-party dependencies
import numpy as np
import tensorflow as tf

import time


def image_preparation_worker(batch_size: int,
                             request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             model_path: str):
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
    model_path : str
        Path to the model.

    Side Effects
    ------------
    - Logs various messages regarding the image preparation status.
    """

    logger = logging.getLogger(__name__)
    logger.info("Image Preparation Worker process started")

    # Disable GPU visibility to prevent memory allocation issues
    tf.config.set_visible_devices([], 'GPU')

    # Load the saved model to get the input tensor shape
    model = tf.saved_model.load(model_path)

    # Get the concrete function from the saved model
    concrete_func = model.signatures[tf.saved_model
                                     .DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # Get the input tensor shape
    num_channels = concrete_func.inputs[0].shape.as_list()[-1]
    logger.debug(f"Input tensor shape: {concrete_func.inputs[0].shape}")

    # Define the maximum time to wait for new images
    TIMEOUT_DURATION = 1
    MAX_WAIT_COUNT = 1

    wait_count = 0

    try:
        while True:
            batch_images, batch_groups, batch_identifiers = [], [], []

            while len(batch_images) < batch_size:
                try:
                    image, group, identifier = request_queue.get(
                        timeout=TIMEOUT_DURATION)
                    logger.debug(f"Retrieved {identifier} from request_queue")

                    image = prepare_image(identifier, image, num_channels)

                    logger.debug(
                        f"Prepared image {identifier} with shape: "
                        f"{image.shape}")

                    batch_images.append(image)
                    batch_groups.append(group)
                    batch_identifiers.append(identifier)

                    request_queue.task_done()
                    wait_count = 0
                except Empty:
                    wait_count += 1
                    logger.debug(
                        "Time without new images: "
                        f"{wait_count * TIMEOUT_DURATION} seconds")

                    if wait_count > MAX_WAIT_COUNT and len(batch_images) > 0:
                        break

            # Determine the maximum width among all images in the batch
            max_width = max(image.shape[0] for image in batch_images)

            # Resize each image in the batch to the maximum width
            for i in range(len(batch_images)):
                batch_images[i] = center_pad_to_width(
                    batch_images[i], max_width, -10)

            logger.info(f"Prepared batch of {len(batch_images)} images")

            # Push the prepared batch to the prepared_queue
            prepared_queue.put(
                (np.array(batch_images), batch_groups, batch_identifiers))
            logger.debug("Pushed prepared batch to prepared_queue")
            logger.debug(
                f"{request_queue.qsize()} images waiting to be processed")
            logger.debug(
                f"{prepared_queue.qsize()} batches ready for prediction")

    except KeyboardInterrupt:
        logger.warning(
            "Image Preparation Worker process interrupted. Exiting...")
    except Exception as e:
        logger.error(f"Error: {e}")


def center_pad_to_width(image: tf.Tensor, target_width: int, pad_value: float):
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

    # Calculate the padding sizes
    total_pad_width = target_width - current_width
    pad_top = total_pad_width // 2
    pad_bottom = total_pad_width - pad_top

    # Configure padding
    padding = [[pad_top, pad_bottom], [0, 0], [0, 0]]

    # Pad the image
    return tf.pad(image, padding, "CONSTANT", constant_values=pad_value)


def prepare_image(identifier: str,
                  image_bytes: bytes,
                  num_channels: int) -> tf.Tensor:
    """
    Prepare a raw image for batch processing.

    Decodes, resizes, normalizes, pads, and transposes the image for further
    processing.

    Parameters
    ----------
    identifier : str
        Identifier of the image (used for logging).
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

    # Normalize the image and something else
    image = 0.5 - (image / 255)

    # Transpose the image
    image = tf.transpose(image, perm=[1, 0, 2])

    return image
