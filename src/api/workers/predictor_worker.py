# Imports

# > Standard library
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# > Local Dependencies
from ..dataset_utils import PredictionDatasetBuilder

# Correct sys.path modification for worker context
current_worker_file_dir = os.path.dirname(os.path.realpath(__file__))
api_dir = os.path.dirname(current_worker_file_dir)
src_dir = os.path.dirname(api_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# > Local imports
from model.management import load_model_from_directory
from setup.environment import initialize_strategy, setup_gpus

logger = logging.getLogger(__name__)


def get_model_channels(config_path: str) -> int:
    """
    Determine the number of input channels from the model config.

    Parameters
    ----------
    config_path : str
        Path to the model directory containing 'config.json'.

    Returns
    -------
    int
        Number of input channels for the model.
    """
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
    """
    Load a Keras model using a distribution strategy.

    Parameters
    ----------
    base_model_dir : str
        Path to base directory containing model folders.
    model_name : str
        Name of the model to load.
    strategy : tf.distribute.Strategy
        Distribution strategy to use when loading the model.

    Returns
    -------
    Tuple[tf.keras.Model, int]
        The loaded model and number of channels it expects.
    """
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


def _output_prediction_error(
    group_id_tensor: tf.Tensor,
    identifier_tensor: tf.Tensor,
    error_text: str,
):
    """
    Logs a prediction error associated with a specific image.

    Parameters
    ----------
    group_id_tensor : tf.Tensor
        Tensor containing the group ID.
    identifier_tensor : tf.Tensor
        Tensor containing the identifier.
    error_text : str
        Description of the error.
    """
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
    images: tf.Tensor,
    groups: tf.Tensor,
    identifiers: tf.Tensor,
    batch_id: str,
    min_chunk: int = 1,
) -> Optional[np.ndarray]:
    """
    Attempts model prediction, retries on OOM by recursively splitting.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded Keras model.
    images : tf.Tensor
        Tensor of images to predict on.
    groups : tf.Tensor
        Tensor of group IDs.
    identifiers : tf.Tensor
        Tensor of image IDs.
    batch_id : str
        Batch identifier for logging.
    min_chunk : int, optional
        Minimum chunk size to pass

    Returns
    -------
    Optional[np.ndarray]
        Numpy array of predictions or None if prediction fails.
    """
    stack = [(images, groups, identifiers, batch_id)]
    results: List[np.ndarray] = []

    while stack:
        imgs, grps, ids, bid = stack.pop()
        try:
            results.append(model.predict_on_batch(imgs))
        except tf.errors.ResourceExhaustedError as oom:
            n = int(imgs.shape[0])  # static or eager-known
            if n <= min_chunk:
                _output_prediction_error(grps[0], ids[0], f"OOM: {oom}")
                continue
            mid = n // 2
            stack.append((imgs[mid:], grps[mid:], ids[mid:], f"{bid}-B"))
            stack.append((imgs[:mid], grps[:mid], ids[:mid], f"{bid}-A"))
        except Exception as exc:
            logger.error("Batch %s failed: %s", bid, exc, exc_info=True)
            for g, i in zip(grps, ids):
                _output_prediction_error(g, i, f"Predict error: {exc}")

    if not results:
        return None
    return np.concatenate(results)


def _send_predictions(
    out_queue: Any,
    preds: np.ndarray,
    groups: tf.Tensor,
    identifiers: tf.Tensor,
    model_name: str,
    batch_uuid: str,
    whitelists: tf.Tensor,
    unique_keys: tf.Tensor,
):
    try:
        out_queue.put(
            (
                preds,
                groups,
                identifiers,
                model_name,
                batch_uuid,
                whitelists,
                unique_keys,
            ),
            timeout=10.0,
        )
    except Exception as e:
        logger.error("Predicted‑batch queue error for %s: %s", batch_uuid, e)


def _run_prediction_loop(
    model: tf.keras.Model,
    num_channels: int,
    current_model_holder: List[str],
    dataset_builder: PredictionDatasetBuilder,
    out_queue: Any,
    stop_event: Any,
    strategy: tf.distribute.Strategy,
):
    total_preds = 0
    batches_count = 0

    while not stop_event.is_set():
        active_model_name_for_loop = current_model_holder[0]
        dataset = dataset_builder.build_tf_dataset()

        for batch_data in dataset:
            if stop_event.is_set():
                break

            (
                images_tensor,
                groups_tensor,
                ids_tensor,
                model_paths_tensor,
                whitelists_tensor,
                unique_keys_tensor,
            ) = batch_data

            # Filter out padding (where identifier is empty string)
            item_mask = tf.not_equal(ids_tensor, "")
            if not tf.reduce_any(item_mask):
                logger.debug("Batch was all padding. Skipping.")
                continue

            images = tf.boolean_mask(images_tensor, item_mask)
            groups = tf.boolean_mask(groups_tensor, item_mask)
            identifiers = tf.boolean_mask(ids_tensor, item_mask)
            whitelists = tf.boolean_mask(whitelists_tensor, item_mask)
            unique_keys = tf.boolean_mask(unique_keys_tensor, item_mask)

            if tf.shape(images)[0] == 0:
                continue

            batch_uuid = str(uuid.uuid4())
            logger.debug(
                "Predicting batch %s (n=%d) with model %s",
                batch_uuid,
                tf.shape(images)[0],
                active_model_name_for_loop,
            )

            start_time = time.time()
            preds_np = _safe_predict(
                model,
                images,
                groups,
                identifiers,
                batch_uuid,
            )
            duration = time.time() - start_time

            if preds_np is None:
                logger.warning("No predictions for batch %s", batch_uuid)
                continue

            total_preds += len(preds_np)
            batches_count += 1

            logger.info(
                "Predicted %d items in %.2fs (total=%d, batches=%d)",
                len(preds_np),
                duration,
                total_preds,
                batches_count,
            )

            _send_predictions(
                out_queue,
                preds_np,
                groups,
                identifiers,
                active_model_name_for_loop,
                batch_uuid,
                whitelists,
                unique_keys,
            )

        # Check if model name changed during dataset iteration
        if (
            current_model_holder[0] != active_model_name_for_loop
            and not stop_event.is_set()
        ):
            logger.info(
                "Switching model: %s → %s",
                active_model_name_for_loop,
                current_model_holder[0],
            )
            try:
                model, num_channels = create_model_with_strategy(
                    dataset_builder.base_dir,
                    current_model_holder[0],
                    strategy,
                )
                dataset_builder.num_channels = num_channels
            except Exception as e:
                logger.critical(
                    f"Failed to load new model {current_model_holder[0]}: {e}. Worker stopping.",
                    exc_info=True,
                )
                stop_event.set()
                break

        logger.info("Prediction loop stopped (total predictions=%d)", total_preds)


def predictor_process_entrypoint(
    mp_request_queue: mp.Queue,
    mp_predicted_batches_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
):
    """
    Main entry point for the predictor worker process.

    Parameters
    ----------
    mp_request_queue : mp.Queue
        Queue from which image requests are read.
    mp_predicted_batches_queue : mp.Queue
        Queue to send prediction results to.
    config : Dict[str, Any]
        Dictionary containing model config, GPU usage, batch size, etc.
    stop_event : mp.Event
        Multiprocessing event to indicate shutdown.
    """
    worker_config = {
        "base_model_dir": config["base_model_dir"],
        "initial_model_name": config["model_name"],
        "gpus": config["gpus"],
        "batch_size": config["batch_size"],
        "patience": config["patience"],
    }
    logger.info(f"Predictor worker starting with config: {worker_config}")

    active_gpus = setup_gpus(worker_config["gpus"])
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

    dataset_builder = PredictionDatasetBuilder(
        mp_request_queue,
        worker_config["batch_size"],
        current_model_name_holder,
        num_channels,
        stop_event,
        worker_config["patience"],
    )

    _run_prediction_loop(
        model,
        num_channels,
        current_model_name_holder,
        dataset_builder,
        mp_predicted_batches_queue,
        stop_event,
        strategy,
    )
