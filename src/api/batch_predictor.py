# Imports

# > Standard library
import logging
import multiprocessing
import json
import os
import sys
import time
import uuid

from typing import List, Tuple, Optional

# > Third-party dependencies
import numpy as np
import tensorflow as tf
from multiprocessing.queues import Full  # For type hinting


# Add parent directory to path for imports
from callbacks import attempt_callback

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
    logger = logging.getLogger(__name__)
    try:
        gpu_devices = tf.config.list_physical_devices("GPU")
        logger.info("Available GPUs: %s", gpu_devices)

        if gpus == "-1":  # CPU only
            active_gpus = []
            tf.config.set_visible_devices([], "GPU")
            logger.info("Using CPU only as per configuration.")
        elif gpus.lower() == "all":
            active_gpus = gpu_devices
            tf.config.set_visible_devices(active_gpus, "GPU")
            logger.info("Using all available GPUs: %s", active_gpus)
        else:
            gpu_indices_str = gpus.split(",")
            chosen_gpus = []
            for idx_str in gpu_indices_str:
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(gpu_devices):
                        chosen_gpus.append(gpu_devices[idx])
                    else:
                        logger.warning(
                            f"GPU index {idx} is out of range. Max index is {len(gpu_devices) - 1}"
                        )
                except ValueError:
                    logger.warning(f"Invalid GPU index: {idx_str}. Must be an integer.")
            active_gpus = chosen_gpus
            tf.config.set_visible_devices(active_gpus, "GPU")
            logger.info("Using specific GPU(s): %s", active_gpus)

        for gpu in active_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Memory growth enabled for {gpu.name}")

    except Exception as e:
        logger.error(f"Error setting up GPU environment: {e}. Falling back to CPU.")
        tf.config.set_visible_devices([], "GPU")  # Fallback to CPU
        active_gpus = []

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
    actual_config_path = os.path.join(config_path, "config.json")
    logger = logging.getLogger(__name__)

    if not os.path.exists(actual_config_path):
        raise FileNotFoundError(
            f"Config file not found in the directory: {actual_config_path}"
        )

    # Load the configuration file
    with open(actual_config_path, "r", encoding="UTF-8") as file:
        config = json.load(file)

    # Extract the number of channels
    # First, check the "model_channels" key, then the "args" key
    num_channels = config.get(
        "model_channels", config.get("args", {}).get("channels", None)
    )
    if num_channels is None:
        # Fallback for older configs or different structures if necessary
        num_channels = config.get("input_shape", [None, None, None, None])[-1]
        if num_channels is None:
            # Default to 1 if not found, or raise error
            logger.warning("Number of channels not found in config, defaulting to 1.")
            num_channels = 1  # Or raise ValueError("Number of channels not found in the config file.")

    logger.debug("Number of channels retrieved for %s: %d", config_path, num_channels)
    return int(num_channels)


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
    """
    logger = logging.getLogger(__name__)
    try:
        num_channels = get_model_channels(model_path)
        logger.debug("New number of channels for %s: %s", model_path, num_channels)
        return num_channels
    except Exception as e:
        logger.error("Error updating channels for %s: %s", model_path, e)
        raise e


def create_model(
    base_model_dir: str, model_path: str, strategy: tf.distribute.Strategy
) -> (tf.keras.Model, int):
    """
    Load a pre-trained TensorFlow model within the given distribution strategy scope.
    """
    logger = logging.getLogger(__name__)
    with strategy.scope():
        try:
            model_location = os.path.join(base_model_dir, model_path)
            if not os.path.isdir(model_location):
                raise FileNotFoundError(f"Model directory not found: {model_location}")
            model = load_model_from_directory(model_location, compile=False)
            num_channels = update_channels(model_location)  # This uses model_location
            logger.info(
                "Model '%s' loaded successfully from '%s' with %d channels.",
                model.name,
                model_location,
                num_channels,
            )
        except Exception as e:
            logger.error(
                "Error loading model from '%s' (resolved: %s): %s",
                model_path,
                model_location,
                e,
                exc_info=True,
            )
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
    mp_request_queue: multiprocessing.Queue,  # Changed from request_queue
    current_model_path_holder: list,  # Stores [current_model_path_str]
    stop_event: multiprocessing.Event,
    patience: float,  # Changed from int to float
):
    """
    Generator that yields data from the mp_request_queue.
    It also handles model switching: if an item in the queue specifies a different model
    than current_model_path_holder[0], it puts the item back and signals a model change
    by breaking, so the dataset can be re-created.
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        f"New data generator started for model: {current_model_path_holder[0]}. Patience: {patience}s"
    )
    time_since_last_item_processed = time.time()
    items_in_current_batch = 0

    while not stop_event.is_set():
        try:
            # Get item from MP queue (fed by async bridge)
            # Timeout allows checking stop_event and patience logic
            item_tuple = mp_request_queue.get(
                timeout=0.1
            )  # (image_bytes, group_id, identifier, model_path_str, whitelist_list)
            if item_tuple is None:  # Sentinel
                logger.info("Data generator received sentinel. Stopping.")
                break

            items_in_current_batch += 1
            time_since_last_item_processed = time.time()

            # Check for model change request
            _image_bytes, _group_id, _identifier, item_model_path, _whitelist = (
                item_tuple
            )

            # Ensure current_model_path_holder[0] is initialized if it's the first item
            if current_model_path_holder[0] is None and item_model_path is not None:
                current_model_path_holder[0] = item_model_path
                logger.info(f"Data generator initialized with model: {item_model_path}")

            if (
                item_model_path is not None
                and item_model_path != current_model_path_holder[0]
            ):
                logger.info(
                    f"Model change detected in data_generator. Current: '{current_model_path_holder[0]}', "
                    f"Requested: '{item_model_path}'. Pushing item back and stopping generator for model switch."
                )
                # Put item back to be processed by the *next* data_generator for the new model
                try:
                    mp_request_queue.put(
                        item_tuple, timeout=1.0
                    )  # Put it back carefully
                except Full:
                    logger.error(
                        f"MP Request Queue full while trying to requeue item for model switch. Item for {item_model_path} may be lost."
                    )

                current_model_path_holder[0] = item_model_path  # Signal to outer loop
                break  # Stop this generator instance; dataset will be rebuilt

            yield item_tuple

        except multiprocessing.queues.Empty:
            # Queue is empty, check patience
            if items_in_current_batch > 0 and (
                time.time() - time_since_last_item_processed > patience
            ):
                logger.debug(
                    f"Data generator patience ({patience}s) exceeded with {items_in_current_batch} items. "
                    "Finalizing current batch."
                )
                break  # Stop generator, yield collected batch
            # else: continue loop, waiting for items or stop_event
        except Exception as e:
            logger.error(f"Error in data_generator: {e}", exc_info=True)
            # Decide if to break or continue
            break  # Safer to break on unknown error

    logger.debug(
        f"Data generator for model {current_model_path_holder[0] if current_model_path_holder else 'N/A'} stopped. Yielded {items_in_current_batch} items."
    )


def create_dataset(
    mp_request_queue: multiprocessing.Queue,  # Changed from request_queue
    batch_size: int,
    current_model_path_holder: list,  # Passed through to data_generator
    num_channels: int,  # For process_sample
    stop_event: multiprocessing.Event,
    patience: float,  # Changed from int to float
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from the mp_request_queue.
    """
    logger = logging.getLogger(__name__)
    logger.debug(
        f"Creating TensorFlow dataset for model {current_model_path_holder[0]} with {num_channels} channels."
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            mp_request_queue, current_model_path_holder, stop_event, patience
        ),
        # Output types: (image_bytes, group_id, identifier, model_path_str, whitelist_list_of_strings)
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # image_bytes
            tf.TensorSpec(shape=(), dtype=tf.string),  # group_id
            tf.TensorSpec(shape=(), dtype=tf.string),  # identifier
            tf.TensorSpec(shape=(), dtype=tf.string),  # model_path_str
            tf.TensorSpec(
                shape=(None,), dtype=tf.string
            ),  # whitelist (list of strings)
        ),
    )

    # Map process_sample
    dataset = dataset.map(
        lambda img_bytes, grp_id, idntfr, mdl_path, wl: process_sample(
            img_bytes, grp_id, idntfr, mdl_path, wl, num_channels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,  # Keep autotune
        deterministic=False,  # Keep false for performance unless order is critical
    )

    # Padded batching
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        # Shapes after process_sample: (image (W,H,C), group_id (), identifier (), model_path_str (), whitelist (N,))
        padded_shapes=(
            [
                None,
                None,
                num_channels,
            ],  # image (variable width, fixed height 64, num_channels)
            [],  # group_id scalar string
            [],  # identifier scalar string
            [],  # model_path_str scalar string
            [None],  # whitelist (variable number of strings)
        ),
        padding_values=(
            tf.constant(
                -10.0, dtype=tf.float32
            ),  # Image padding (Loghi specific for `0.5 - image`)
            tf.constant("", dtype=tf.string),  # group_id padding
            tf.constant("", dtype=tf.string),  # identifier padding
            tf.constant(
                "", dtype=tf.string
            ),  # model_path_str padding for empty batch slots
            tf.constant("", dtype=tf.string),  # whitelist string padding
        ),
        drop_remainder=False,  # Process all items, even if last batch is smaller
    )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    logger.debug("TensorFlow dataset created successfully.")
    return dataset


def output_prediction_error(
    group_id_tensor: tf.Tensor,
    identifier_tensor: tf.Tensor,
    error_text: str,
    callback_url: str,
):
    """
    Send an error callback for a specific prediction.
    """
    logger = logging.getLogger(__name__)
    try:
        group_id = group_id_tensor.numpy().decode("utf-8", errors="ignore")
        identifier = identifier_tensor.numpy().decode("utf-8", errors="ignore")

        if not identifier and not group_id:  # Skip if it's a padding artifact
            return

        logger.error(f"Prediction error for {group_id}/{identifier}: {error_text}")
        payload = {
            "group_id": group_id,
            "identifier": identifier,
            "status": "predict ERROR",
            "result": str(error_text),  # Ensure it's a string
        }
        if callback_url and callback_url.lower() != "none":
            attempt_callback(callback_url, payload)  # This is synchronous
        else:
            logger.debug("Callback URL not set, skipping error callback.")

    except Exception as e:
        logger.error(f"Failed to send error callback or decode ids: {e}")


def predict_on_batch_with_tf_function(
    model: tf.keras.Model, batch_images: tf.Tensor
) -> np.ndarray:
    """
    Make predictions on a batch of images using the provided model, wrapped in tf.function.
    """

    # This internal tf.function can improve performance by building a graph
    # It's crucial that model.predict_on_batch itself is graph-compatible or this won't help much
    # For many Keras models, predict_on_batch uses underlying tf.function-decorated methods.
    @tf.function
    def _predict(images):
        return model(
            images, training=False
        )  # Using model call directly is often better for tf.function

    return _predict(batch_images).numpy()  # .numpy() to get back to Python/Numpy land


def safe_predict(
    model: tf.keras.Model,
    batch_images: tf.Tensor,
    # batch_info: List[Tuple[tf.Tensor, tf.Tensor, str]], # (group_id_tensor, identifier_tensor, model_path_str)
    # The model_path_str is not needed here, but group_id and identifier are for error reporting
    batch_groups: tf.Tensor,  # Tensor of group_id strings
    batch_identifiers: tf.Tensor,  # Tensor of identifier strings
    batch_id_str: str,  # For logging
    callback_url: str,
) -> Optional[np.ndarray]:  # Return Optional np.ndarray, None if all fail
    """
    Attempt to predict on a batch of images. Handles OOM by splitting.
    """
    logger = logging.getLogger(__name__)
    try:
        # Using model.predict_on_batch as it's standard for batch processing in Keras
        # If issues, `predict_on_batch_with_tf_function` can be tried.
        # However, `model.predict_on_batch` is generally efficient.
        # For single replica / non-distributed, model(batch_images, training=False) is also common.
        encoded_predictions = model.predict_on_batch(batch_images)
        return encoded_predictions

    except tf.errors.ResourceExhaustedError as e:
        logger.warning(
            f"OOM error with batch size {len(batch_images)} (Batch ID: {batch_id_str}). Details: {e}. Splitting and retrying."
        )
        if len(batch_images) == 1:
            logger.error(
                f"OOM error with single image (Batch ID: {batch_id_str}). Skipping image: "
                f"Group: {batch_groups[0].numpy().decode('utf-8', 'ignore')}, "
                f"ID: {batch_identifiers[0].numpy().decode('utf-8', 'ignore')}."
            )
            output_prediction_error(
                batch_groups[0], batch_identifiers[0], f"OOM error: {e}", callback_url
            )
            return None  # No prediction for this single image

        mid_index = len(batch_images) // 2
        first_half_preds = safe_predict(
            model,
            batch_images[:mid_index],
            batch_groups[:mid_index],
            batch_identifiers[:mid_index],
            f"{batch_id_str}-A",
            callback_url,
        )
        second_half_preds = safe_predict(
            model,
            batch_images[mid_index:],
            batch_groups[mid_index:],
            batch_identifiers[mid_index:],
            f"{batch_id_str}-B",
            callback_url,
        )

        if first_half_preds is not None and second_half_preds is not None:
            return np.concatenate((first_half_preds, second_half_preds))
        elif first_half_preds is not None:
            return first_half_preds
        elif second_half_preds is not None:
            return second_half_preds
        else:
            return None  # Both halves failed

    except Exception as e:
        logger.error(
            f"Unexpected error predicting batch {batch_id_str}: {e}", exc_info=True
        )
        for i in range(len(batch_images)):
            output_prediction_error(
                batch_groups[i],
                batch_identifiers[i],
                f"Prediction error: {e}",
                callback_url,
            )
        return None  # No predictions for this batch due to unexpected error


def batch_prediction_worker(
    mp_request_queue: multiprocessing.Queue,  # Input: From async bridge
    mp_predicted_batches_queue: multiprocessing.Queue,  # Output: To decoding worker
    base_model_dir: str,
    initial_model_path: str,  # Relative to base_model_dir
    error_output_path: str,  # For any specific error files (less used now)
    stop_event: multiprocessing.Event,
    callback_url: str,  # For OOM errors from safe_predict
    gpus: str = "0",
    batch_size: int = 32,
    patience: float = 1.0,  # How long data_generator waits for more items
):
    """
    Worker process for performing batch predictions on images.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Batch prediction worker started. Initial model: {initial_model_path}, GPU config: {gpus}"
    )

    active_gpus = setup_gpu_environment(gpus)  # Sets visible devices
    strategy = initialize_strategy(
        use_float32=False, active_gpus=active_gpus
    )  # Or True based on model needs

    current_model_path_holder = [
        initial_model_path
    ]  # Mutable list for data_generator to update on model switch

    try:
        model, num_channels = create_model(
            base_model_dir, current_model_path_holder[0], strategy
        )
    except Exception as e:
        logger.error(
            f"Failed to load initial model {initial_model_path}: {e}. Prediction worker cannot start.",
            exc_info=True,
        )
        return  # Cannot proceed without a model

    total_predictions_made = 0
    batches_processed = 0

    while not stop_event.is_set():
        # Create dataset. current_model_path_holder[0] might be updated by data_generator if a model switch occurs
        # The dataset generation will then stop, and this loop will re-create it with the new model path.
        previous_model_path_for_loop = current_model_path_holder[0]

        dataset = create_dataset(
            mp_request_queue,
            batch_size,
            current_model_path_holder,  # Passed to data_generator
            num_channels,
            stop_event,
            patience,
        )

        batch_iterator = iter(dataset)

        for batch_data in batch_iterator:
            if stop_event.is_set():
                logger.info("Stop event set, breaking from batch processing loop.")
                break

            # batch_data is (images, batch_groups, batch_identifiers, batch_model_paths, batch_whitelists)
            # All items in batch_model_paths should be the same due to data_generator logic, or empty for padding.
            # We use current_model_path_holder[0] as the authoritative model for this dataset iteration.

            (
                images_tensor,
                groups_tensor,
                ids_tensor,
                model_paths_tensor,
                whitelists_tensor,
            ) = batch_data

            # Filter out padding: if identifier is empty string, it's likely padding
            actual_item_mask = tf.not_equal(ids_tensor, "")
            if not tf.reduce_any(actual_item_mask):
                logger.debug("Empty batch after filtering padding, skipping.")
                continue  # All items were padding

            # Apply mask to all tensors
            actual_images = tf.boolean_mask(images_tensor, actual_item_mask)
            actual_groups = tf.boolean_mask(groups_tensor, actual_item_mask)
            actual_ids = tf.boolean_mask(ids_tensor, actual_item_mask)
            actual_model_paths = tf.boolean_mask(
                model_paths_tensor, actual_item_mask
            )  # Should all be same or empty
            actual_whitelists = tf.boolean_mask(
                whitelists_tensor, actual_item_mask
            )  # This is a RaggedTensor, boolean_mask works

            if tf.shape(actual_images)[0] == 0:  # Double check after masking
                logger.debug(
                    "Batch empty after filtering padding (second check), skipping."
                )
                continue

            # Perform predictions
            batch_process_id = str(
                uuid.uuid4()
            )  # Unique ID for this processing attempt

            logger.info(
                f"Predicting batch of size {tf.shape(actual_images)[0]} (ID: {batch_process_id}) with model {current_model_path_holder[0]}"
            )
            t_predict_start = time.time()

            encoded_predictions_np = safe_predict(
                model,
                actual_images,
                actual_groups,  # For error reporting in safe_predict
                actual_ids,  # For error reporting in safe_predict
                batch_process_id,
                callback_url,  # For OOM errors
            )

            t_predict_duration = time.time() - t_predict_start

            if encoded_predictions_np is not None and len(encoded_predictions_np) > 0:
                num_preds = len(encoded_predictions_np)
                total_predictions_made += num_preds
                batches_processed += 1
                logger.info(
                    f"Made {num_preds} predictions in {t_predict_duration:.2f}s (Batch ID: {batch_process_id}). "
                    f"Total predictions: {total_predictions_made}. Batches: {batches_processed}."
                )
                try:
                    # Send (predictions, groups, ids, model_path_used, batch_id, whitelists)
                    # Ensure all are serializable (e.g., numpy arrays for tensors if needed by mp.Queue)
                    # batch_groups, batch_identifiers, batch_whitelists are already tf.Tensor here.
                    # The mp.Queue should handle them, but receiver (decoder) must expect tf.Tensor.
                    mp_predicted_batches_queue.put(
                        (
                            encoded_predictions_np,  # This is already np.ndarray
                            actual_groups,  # tf.Tensor (strings)
                            actual_ids,  # tf.Tensor (strings)
                            current_model_path_holder[
                                0
                            ],  # Model path used for this prediction
                            batch_process_id,  # For tracking
                            actual_whitelists,  # tf.RaggedTensor / tf.Tensor (strings)
                        ),
                        timeout=10.0,  # Add a timeout to prevent indefinite blocking
                    )
                except Full:
                    logger.error(
                        f"Predicted batches queue is full. Predictions for batch {batch_process_id} may be lost."
                    )
                except Exception as e:
                    logger.error(
                        f"Error putting predictions to queue for batch {batch_process_id}: {e}"
                    )
            elif (
                encoded_predictions_np is not None and len(encoded_predictions_np) == 0
            ):
                logger.warning(
                    f"safe_predict returned empty predictions for batch {batch_process_id}. Not queueing."
                )
            else:  # encoded_predictions_np is None
                logger.warning(
                    f"safe_predict failed for batch {batch_process_id} (returned None). Not queueing."
                )

        # After iterating through dataset, check if model path changed by data_generator
        if stop_event.is_set():
            logger.info(
                "Stop event detected after dataset iteration. Exiting prediction worker."
            )
            break  # Exit main while loop

        if current_model_path_holder[0] != previous_model_path_for_loop:
            logger.info(
                f"Model switch detected by prediction worker. Old: '{previous_model_path_for_loop}', "
                f"New: '{current_model_path_holder[0]}'. Reloading model."
            )
            try:
                model, num_channels = create_model(
                    base_model_dir, current_model_path_holder[0], strategy
                )
                logger.info(
                    f"Successfully switched to model: {current_model_path_holder[0]}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load new model {current_model_path_holder[0]}: {e}. Stopping worker.",
                    exc_info=True,
                )
                stop_event.set()  # Signal to stop everything if model load fails
                break
        # else: model path did not change, or data_generator yielded nothing, continue to make new dataset

    logger.info(
        f"Batch prediction worker stopped. Total predictions made: {total_predictions_made}. Batches processed: {batches_processed}."
    )
