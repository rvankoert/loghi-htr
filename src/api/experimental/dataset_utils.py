# Imports

# > Standard Library
import logging
import multiprocessing as mp
import time
from typing import Any, Generator, List, Tuple

# > Third-party Dependencies
import tensorflow as tf

logger = logging.getLogger(__name__)


def _process_sample_tf(
    image_bytes: tf.Tensor,
    group_id: tf.Tensor,
    identifier: tf.Tensor,
    model_path_tensor: tf.Tensor,
    whitelist: tf.Tensor,
    unique_request_key: tf.Tensor,
    num_channels: int,
) -> tuple:
    """
    Convert a raw image tensor and metadata into a preprocessed sample.

    Parameters
    ----------
    image_bytes : tf.Tensor
        Raw bytes of the image.
    group_id : tf.Tensor
        Group identifier.
    identifier : tf.Tensor
        Unique image ID.
    model_path_tensor : tf.Tensor
        Tensor containing model path string.
    whitelist : tf.Tensor
        Allowed metadata keys.
    unique_request_key : tf.Tensor
        Unique key for the request.
    num_channels : int
        Number of channels the model expects.

    Returns
    -------
    tuple
        Tuple of processed tensors: (image, group_id, identifier, model_path_tensor, whitelist, unique_request_key).
    """
    try:
        image = tf.io.decode_image(
            image_bytes, channels=num_channels, expand_animations=False
        )
    except tf.errors.InvalidArgumentError:  # Handle problematic images
        logger.error(
            f"Invalid image for identifier: {identifier.numpy().decode('utf-8', 'ignore')}. Using zero tensor."
        )
        image = tf.zeros([64, 64, num_channels], dtype=tf.uint8)

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

    # Transpose the image
    image = tf.transpose(image, perm=[1, 0, 2])  # HWC to WHC

    return image, group_id, identifier, model_path_tensor, whitelist, unique_request_key


class PredictionDatasetBuilder:
    """
    Utility to convert a multiprocessing queue into a TensorFlow dataset for prediction.

    Attributes
    ----------
    mp_request_queue : mp.Queue
        Incoming data queue.
    batch_size : int
        Size of each batch.
    current_model_path_holder : List[str]
        Mutable reference holding the current model path.
    num_channels : int
        Number of channels used in preprocessing.
    stop_event : mp.Event
        Shared event used to stop the worker.
    patience : float
        Seconds to wait before yielding batch and stopping on inactivity.
    """

    def __init__(
        self,
        mp_request_queue: mp.Queue,
        batch_size: int,
        current_model_path_holder: List[str],
        num_channels: int,
        stop_event: mp.Event,
        patience: float,
        base_dir: str,
    ):
        self._q = mp_request_queue
        self.batch_size = batch_size
        self.model_path_holder = current_model_path_holder
        self.num_channels = num_channels
        self._stop = stop_event
        self.patience = patience
        self.base_dir = base_dir

    def _poll_queue(self, timeout: float = 0.01) -> Any | None:
        """Return next item or *None* if queue is empty within *timeout*."""
        try:
            return self._q.get(timeout=timeout)
        except mp.queues.Empty:  # type: ignore[attr-defined]
            return None

    def _handle_model_switch(self, sample: Tuple[Any, ...]) -> bool:
        """Return *True* if builder should break and rebuild for a new model."""
        sample_model = sample[3]
        current = self.model_path_holder[0]
        if current is None:
            self.model_path_holder[0] = sample_model
            return False
        if sample_model is not None and sample_model != current:
            # push back and signal switch
            self._q.put(sample)
            self.model_path_holder[0] = sample_model
            logger.info("Model changed → %s; rebuilding dataset", sample_model)
            return True
        return False

    def _data_generator(self) -> Generator[Tuple[Any, ...], None, None]:
        idle_since = time.monotonic()
        had_data = False

        while not self._stop.is_set():
            sample = self._poll_queue()

            if sample is None:
                # No item retrieved in this poll‑interval
                if had_data and (time.monotonic() - idle_since > self.patience):
                    logger.debug(
                        "No new requests for %.1fs – generator exiting",
                        self.patience,
                    )
                    break
                continue  # keep polling

            # Got data
            had_data = True
            idle_since = time.monotonic()

            if self._handle_model_switch(sample):
                break  # outer loop will rebuild for new model

            yield sample

        logging.debug("Data generator stopped")

    def build_tf_dataset(self) -> tf.data.Dataset:
        logger.debug("Building dataset for model %s", self.model_path_holder[0])
        ds = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(
                tf.TensorSpec([], tf.string),  # image_bytes
                tf.TensorSpec([], tf.string),  # group_id
                tf.TensorSpec([], tf.string),  # identifier
                tf.TensorSpec([], tf.string),  # model_path_str
                tf.TensorSpec([None], tf.string),  # whitelist
                tf.TensorSpec([], tf.string),  # unique_request_key
            ),
        )
        ds = ds.map(
            lambda *x: _process_sample_tf(*x, self.num_channels),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        ds = ds.padded_batch(
            self.batch_size,
            padded_shapes=([None, None, self.num_channels], [], [], [], [None], []),
            padding_values=(
                tf.constant(-10.0, tf.float32),
                *(tf.constant("", tf.string) for _ in range(5)),
            ),
            drop_remainder=False,
        ).prefetch(tf.data.AUTOTUNE)
        return ds
