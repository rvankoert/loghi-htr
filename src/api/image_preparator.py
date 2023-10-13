# Imports

# > Standard library
import logging
import multiprocessing

# > Third-party dependencies
import tensorflow as tf


def image_preparation_worker(request_queue: multiprocessing.Queue,
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

    try:
        while True:
            image, group, identifier = request_queue.get()
            logger.debug(f"Retrieved {identifier} from request_queue")

            image = prepare_image(identifier, image, num_channels)

            logger.debug(
                f"Prepared image {identifier} with shape: {image.shape}")

            # Push the prepared image to the prepared_queue
            prepared_queue.put((image.numpy(), group, identifier))
            logger.debug(
                f"Pushed prepared image {identifier} to prepared_queue")
            logger.debug(
                f"{request_queue.qsize()} images waiting to be processed")

            # Indicate that the task is done
            request_queue.task_done()

            logger.debug(f"Finished processing image {identifier}")
            logger.debug(f"{prepared_queue.qsize()} images ready for batch")

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

    with tf.device('/cpu:0'):
        image = tf.io.decode_jpeg(image_bytes, channels=num_channels)

        # Resize while preserving aspect ratio
        target_height = 64
        image = tf.image.resize(image,
                                [target_height,
                                 tf.cast(target_height * tf.shape(image)[1]
                                         / tf.shape(image)[0], tf.int32)],
                                preserve_aspect_ratio=True)

        # Normalize the image and something else
        image = 0.5 - (image / 255)

        # Pad the image
        image = tf.image.resize_with_pad(
            image, target_height, tf.shape(image)[1] + 50)

        # Transpose the image
        image = tf.transpose(image, perm=[1, 0, 2])

        return image
