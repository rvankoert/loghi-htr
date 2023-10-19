# Imports

# > Standard library
import logging
import multiprocessing
from multiprocessing.queues import Empty

# > Third-party dependencies
import numpy as np
import tensorflow as tf


def image_preparation_worker(batch_size: int,
                             request_queue: multiprocessing.Queue,
                             prepared_queue: multiprocessing.Queue,
                             num_channels: int):
    """
    Worker process to prepare images for batch processing.

    Continuously fetches raw images from the request queue, processes them, and
    pushes the prepared images to the prepared queue.

    Parameters
    ----------
    request_queue : multiprocessing.Queue
        Queue from which raw images are fetched.
    prepared_queue : multiprocessing.Queue
        Queue to which prepared images are pushed.
    num_channels : int
        Number of channels desired for the output image (e.g., 1 for grayscale,
        3 for RGB).

    Side Effects
    ------------
    - Logs various messages regarding the image preparation status.
    """

    logger = logging.getLogger(__name__)
    logger.info("Image Preparation Worker process started")

    # Disable GPU visibility to prevent memory allocation issues
    tf.config.set_visible_devices([], 'GPU')

    TIMEOUT_DURATION = 1
    MAX_WAIT_COUNT = 1

    wait_count = 0

    try:
        while True:
            batch_images, batch_groups, batch_identifiers = [], [], []

            while len(batch_images) < batch_size:
                try:
                    image, group, identifier = request_queue.get(timeout=1)
                    logger.debug(f"Retrieved {identifier} from request_queue")

                    image = prepare_image(identifier, image, num_channels)

                    logger.debug(
                        f"Prepared image {identifier} with shape: "
                        f"{image.shape}")

                    batch_images.append(image.numpy())
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
                batch_images[i] = tf.image.resize_with_pad(
                    batch_images[i], max_width, 64)

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
                            [target_height, target_width],
                            preserve_aspect_ratio=True)

    # Normalize the image and something else
    image = 0.5 - (image / 255)

    # Transpose the image
    image = tf.transpose(image, perm=[1, 0, 2])

    return image
