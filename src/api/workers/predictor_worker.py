# Imports

# > Standard library
import logging
import multiprocessing as mp
import json
import os
import sys
import time
import uuid
from typing import List, Tuple, Optional, Dict, Any

# > Third-party dependencies
import numpy as np
import tensorflow as tf
from multiprocessing.queues import Full as MPQueueFullException


# Correct sys.path modification for worker context
current_worker_file_dir = os.path.dirname(os.path.realpath(__file__))
api_dir = os.path.dirname(current_worker_file_dir)
src_dir = os.path.dirname(api_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# > Local imports (now from correct src/ path)
from model.management import load_model_from_directory
from setup.environment import initialize_strategy

logger = logging.getLogger(__name__)


def setup_gpu_environment(gpus_config: str) -> List[tf.config.PhysicalDevice]:
    """Configure the GPU environment for TensorFlow."""
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        logger.info("Available GPUs: %s", gpu_devices)

        if not gpu_devices:
            logger.info("No GPUs found. Using CPU.")
            tf.config.set_visible_devices([], "GPU")
            return []

        if gpus_config == "-1":  # CPU only
            active_gpus = []
            tf.config.set_visible_devices([], "GPU")
            logger.info("Using CPU only as per configuration.")
        elif gpus_config.lower() == "all":
            active_gpus = gpu_devices
            tf.config.set_visible_devices(active_gpus, "GPU")
            logger.info("Using all available GPUs: %s", active_gpus)
        else:
            gpu_indices_str = gpus_config.split(",")
            chosen_gpus = []
            for idx_str in gpu_indices_str:
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(gpu_devices):
                        chosen_gpus.append(gpu_devices[idx])
                    else:
                        logger.warning(f"GPU index {idx} is out of range.")
                except ValueError:
                    logger.warning(f"Invalid GPU index: {idx_str}.")
            active_gpus = chosen_gpus
            tf.config.set_visible_devices(active_gpus, "GPU")
            logger.info("Using specific GPU(s): %s", active_gpus)

        for gpu in active_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Memory growth enabled for {gpu.name}")
        return active_gpus
    except Exception as e:
        logger.error(f"Error setting up GPU environment: {e}. Falling back to CPU.")
        tf.config.set_visible_devices([], "GPU")
        return []


def get_model_channels(config_path: str) -> int:
    """Retrieve the number of input channels for a model."""
    actual_config_path = os.path.join(config_path, "config.json")
    if not os.path.exists(actual_config_path):
        raise FileNotFoundError(f"Config file not found: {actual_config_path}")
    with open(actual_config_path, "r", encoding="UTF-8") as file:
        config = json.load(file)
    num_channels = config.get("model_channels", config.get("args", {}).get("channels"))
    if num_channels is None:  # Fallback
        num_channels = config.get("input_shape", [None, None, None, 1])[-1]
        num_channels = (
            1 if num_channels is None else num_channels
        )  # Default to 1 if still not found
        logger.warning(
            f"Channels not explicitly found, inferred/defaulted to: {num_channels}"
        )
    return int(num_channels)


def create_model_with_strategy(
    base_model_dir: str, model_name: str, strategy: tf.distribute.Strategy
) -> Tuple[tf.keras.Model, int]:
    """Load a pre-trained TensorFlow model and its channel count."""
    with strategy.scope():
        model_location = os.path.join(base_model_dir, model_name)
        if not os.path.isdir(model_location):
            raise FileNotFoundError(f"Model directory not found: {model_location}")
        model = load_model_from_directory(model_location, compile=False)
        num_channels = get_model_channels(model_location)
        logger.info(
            f"Model '{model.name}' loaded from '{model_location}' with {num_channels} channels."
        )
        return model, num_channels


def _process_sample_tf(
    image_bytes: tf.Tensor,
    group_id: tf.Tensor,
    identifier: tf.Tensor,
    model_path_tensor: tf.Tensor,  # Changed name for clarity
    whitelist: tf.Tensor,
    num_channels: int,
) -> tuple:
    """Preprocess a single sample for the dataset (TensorFlow operations)."""
    try:
        image = tf.io.decode_image(
            image_bytes, channels=num_channels, expand_animations=False
        )
    except tf.errors.InvalidArgumentError:  # Handle problematic images
        logger.error(
            f"Invalid image for identifier: {identifier.numpy().decode('utf-8', 'ignore')}. Using zero tensor."
        )
        image = tf.zeros(
            [64, 64, num_channels], dtype=tf.uint8
        )  # Use uint8 before float conversion

    image = tf.image.resize(
        image, [64, tf.constant(99999, dtype=tf.int32)], preserve_aspect_ratio=True
    )
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize_with_pad(
        image, 64, tf.shape(image)[1] + 50, method=tf.image.ResizeMethod.BILINEAR
    )
    image = 0.5 - image  # Normalize
    image = tf.transpose(image, perm=[1, 0, 2])  # HWC to WHC
    return image, group_id, identifier, model_path_tensor, whitelist


class PredictionDatasetBuilder:
    def __init__(
        self,
        mp_request_queue: mp.Queue,
        batch_size: int,
        current_model_path_holder: List[str],
        num_channels: int,
        stop_event: mp.Event,
        patience: float,
    ):
        self.mp_request_queue = mp_request_queue
        self.batch_size = batch_size
        self.current_model_path_holder = current_model_path_holder
        self.num_channels = num_channels
        self.stop_event = stop_event
        self.patience = patience

    def _data_generator(self):
        """Yields data from mp_request_queue, handles model switching signals."""
        logger.debug(
            f"Dataset generator started for model: {self.current_model_path_holder[0]}. Patience: {self.patience}s. Batch size: {self.batch_size}"
        )
        time_since_last_request = time.time()
        has_data = False

        while not self.stop_event.is_set():
            if not self.mp_request_queue.empty():
                try:
                    time_since_last_request = time.time()
                    data = self.mp_request_queue.get()
                    new_model_path = data[3]

                    if self.current_model_path_holder[0] is None:
                        self.current_model_path_holder[0] = new_model_path

                    if (
                        new_model_path != self.current_model_path_holder[0]
                        and new_model_path is not None
                    ):
                        self.mp_request_queue.put(data)
                        logging.info(
                            "Model changed to '%s'. Switching generator.",
                            new_model_path,
                        )
                        self.current_model_path_holder[0] = new_model_path
                        break  # Exit to allow dataset recreation with the new model

                    has_data = True
                    yield data
                except Exception as e:
                    logging.error("Error retrieving data from queue: %s", e)
            else:
                time.sleep(0.01)  # Prevent busy waiting

                if time.time() - time_since_last_request > self.patience and has_data:
                    logging.debug(
                        "No new requests for %d seconds. Yielding remaining data.",
                        self.patience,
                    )
                    break

        logging.debug("Data generator stopped")

    def build_tf_dataset(self) -> tf.data.Dataset:
        """Creates a TensorFlow dataset."""
        logger.debug(
            f"Building TensorFlow dataset for model {self.current_model_path_holder[0]} with {self.num_channels} channels."
        )
        dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),  # image_bytes
                tf.TensorSpec(shape=(), dtype=tf.string),  # group_id
                tf.TensorSpec(shape=(), dtype=tf.string),  # identifier
                tf.TensorSpec(shape=(), dtype=tf.string),  # model_path_str
                tf.TensorSpec(shape=(None,), dtype=tf.string),  # whitelist
            ),
        )

        dataset = dataset.map(
            lambda img, grp, idf, mdl, wl: _process_sample_tf(
                img, grp, idf, mdl, wl, self.num_channels
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=([None, None, self.num_channels], [], [], [], [None]),
            padding_values=(
                tf.constant(-10.0, dtype=tf.float32),  # Image padding for '0.5 - image'
                tf.constant("", dtype=tf.string),  # group_id
                tf.constant("", dtype=tf.string),  # identifier
                tf.constant("", dtype=tf.string),  # model_path
                tf.constant("", dtype=tf.string),  # whitelist
            ),
            drop_remainder=False,
        )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        logger.debug("TensorFlow dataset built.")
        return dataset


def _output_prediction_error(
    group_id_tensor: tf.Tensor,
    identifier_tensor: tf.Tensor,
    error_text: str,
):
    """Sends an error callback for a specific prediction item."""
    try:
        group_id = group_id_tensor.numpy().decode("utf-8", errors="ignore")
        identifier = identifier_tensor.numpy().decode("utf-8", errors="ignore")
        if not identifier and not group_id:
            return  # Padding

        logger.error(f"Prediction error for {group_id}/{identifier}: {error_text}")
    except Exception as e:
        logger.error(f"Failed to send error callback or decode IDs: {e}")


def _safe_predict(
    model: tf.keras.Model,
    batch_images: tf.Tensor,
    batch_groups: tf.Tensor,
    batch_identifiers: tf.Tensor,
    batch_id_str: str,
) -> Optional[np.ndarray]:
    """Attempts prediction, handles OOM by splitting."""
    try:
        # Using model call directly, often better with tf.function context
        # predictions = model(batch_images, training=False)
        # return predictions.numpy()
        return model.predict_on_batch(batch_images)  # Keras standard, usually efficient
    except tf.errors.ResourceExhaustedError as e:
        logger.warning(
            f"OOM error (batch ID: {batch_id_str}, size {len(batch_images)}). Splitting. Error: {e}"
        )
        if len(batch_images) == 1:
            _output_prediction_error(
                batch_groups[0], batch_identifiers[0], f"OOM error: {e}"
            )
            return None

        mid = len(batch_images) // 2
        preds1 = _safe_predict(
            model,
            batch_images[:mid],
            batch_groups[:mid],
            batch_identifiers[:mid],
            f"{batch_id_str}-A",
        )
        preds2 = _safe_predict(
            model,
            batch_images[mid:],
            batch_groups[mid:],
            batch_identifiers[mid:],
            f"{batch_id_str}-B",
        )

        if preds1 is not None and preds2 is not None:
            return np.concatenate((preds1, preds2))
        if preds1 is not None:
            return preds1
        if preds2 is not None:
            return preds2
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error predicting batch {batch_id_str}: {e}", exc_info=True
        )
        for i in range(len(batch_images)):
            _output_prediction_error(
                batch_groups[i],
                batch_identifiers[i],
                f"Prediction error: {e}",
            )
        return None


def predictor_process_entrypoint(
    mp_request_queue: mp.Queue,
    mp_predicted_batches_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
):
    """Main function for the batch prediction worker process."""
    worker_config = {
        "base_model_dir": config["base_model_dir"],
        "initial_model_name": config["model_name"],
        "gpus": config["gpus"],
        "batch_size": config["batch_size"],
        "patience": config["patience"],
    }
    logger.info(f"Predictor worker starting with config: {worker_config}")

    active_gpus = setup_gpu_environment(worker_config["gpus"])
    strategy = initialize_strategy(use_float32=False, active_gpus=active_gpus)

    # Mutable list to allow data_generator to signal model changes
    current_model_name_holder = [worker_config["initial_model_name"]]

    try:
        model, num_channels = create_model_with_strategy(
            worker_config["base_model_dir"], current_model_name_holder[0], strategy
        )
    except Exception as e:
        logger.critical(
            f"Failed to load initial model {current_model_name_holder[0]}: {e}. Worker exiting.",
            exc_info=True,
        )
        return

    total_preds, batches_count = 0, 0
    dataset_builder = PredictionDatasetBuilder(
        mp_request_queue,
        worker_config["batch_size"],
        current_model_name_holder,  # Will be updated by dataset_builder if model changes
        num_channels,  # Initial num_channels
        stop_event,
        worker_config["patience"],
    )

    while not stop_event.is_set():
        active_model_name_for_loop = current_model_name_holder[0]
        dataset = (
            dataset_builder.build_tf_dataset()
        )  # Uses current_model_name_holder[0] and num_channels

        for (
            batch_data
        ) in dataset:  # Iterates until data_generator stops or patience exceeded
            if stop_event.is_set():
                break

            (
                images_tensor,
                groups_tensor,
                ids_tensor,
                model_paths_tensor,
                whitelists_tensor,
            ) = batch_data

            # Filter out padding (where identifier is empty string)
            actual_item_mask = tf.not_equal(ids_tensor, "")
            if not tf.reduce_any(actual_item_mask):
                logger.debug("Batch was all padding. Skipping.")
                continue

            actual_images = tf.boolean_mask(images_tensor, actual_item_mask)
            actual_groups = tf.boolean_mask(groups_tensor, actual_item_mask)
            actual_ids = tf.boolean_mask(ids_tensor, actual_item_mask)
            # actual_model_paths = tf.boolean_mask(model_paths_tensor, actual_item_mask) # Not strictly needed post-filtering
            actual_whitelists = tf.boolean_mask(
                whitelists_tensor, actual_item_mask
            )  # RaggedTensor if not all same length

            if tf.shape(actual_images)[0] == 0:
                continue  # Should not happen if reduce_any was true

            batch_uuid = str(uuid.uuid4())
            logger.info(
                f"Predicting batch {batch_uuid} (size {tf.shape(actual_images)[0]}) with model {active_model_name_for_loop}"
            )

            start_time = time.time()
            encoded_preds_np = _safe_predict(
                model,
                actual_images,
                actual_groups,
                actual_ids,
                batch_uuid,
            )
            duration = time.time() - start_time

            if encoded_preds_np is not None and len(encoded_preds_np) > 0:
                num_batch_preds = len(encoded_preds_np)
                total_preds += num_batch_preds
                batches_count += 1
                logger.info(
                    f"Predicted {num_batch_preds} items for batch {batch_uuid} in {duration:.2f}s. Total: {total_preds}, Batches: {batches_count}."
                )
                try:
                    mp_predicted_batches_queue.put(
                        (
                            encoded_preds_np,
                            actual_groups,
                            actual_ids,
                            active_model_name_for_loop,
                            batch_uuid,
                            actual_whitelists,
                        ),
                        timeout=10.0,
                    )
                except MPQueueFullException:
                    logger.error(
                        f"Predicted batches queue full. Batch {batch_uuid} lost."
                    )
                except Exception as e:
                    logger.error(f"Error queueing batch {batch_uuid}: {e}")
            else:
                logger.warning(
                    f"No predictions from _safe_predict for batch {batch_uuid}."
                )

        if stop_event.is_set():
            break

        # Check if model name changed during dataset iteration
        if current_model_name_holder[0] != active_model_name_for_loop:
            logger.info(
                f"Model switch detected: from '{active_model_name_for_loop}' to '{current_model_name_holder[0]}'. Reloading."
            )
            try:
                model, num_channels = create_model_with_strategy(
                    worker_config["base_model_dir"],
                    current_model_name_holder[0],
                    strategy,
                )
                dataset_builder.num_channels = (
                    num_channels  # Update num_channels in dataset builder
                )
                logger.info(
                    f"Switched to model: {current_model_name_holder[0]} with {num_channels} channels."
                )
            except Exception as e:
                logger.critical(
                    f"Failed to load new model {current_model_name_holder[0]}: {e}. Worker stopping.",
                    exc_info=True,
                )
                stop_event.set()  # Signal critical failure
                break
        # Else, dataset exhausted (patience) or generator stopped normally. Loop to create new dataset.

    logger.info(
        f"Predictor worker stopped. Total predictions: {total_preds}. Batches processed: {batches_count}."
    )
