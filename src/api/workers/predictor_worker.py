# Imports

# > Standard library
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path
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
from model.management import load_model_from_directory  # noqa: E402
from setup.environment import initialize_strategy, setup_gpus  # noqa: E402

logger = logging.getLogger(__name__)


def get_model_channels(model_dir: str | os.PathLike[str]) -> int:
    """Return the number of input channels expected by the model.

    The channel count is looked‑up in ``config.json`` inside *model_dir*.
    Multiple fallbacks are attempted to maximise robustness.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Directory that contains the model checkpoint and its ``config.json``.

    Returns
    -------
    int
        The number of channels (e.g. ``1`` for grayscale, ``3`` for RGB).

    Raises
    ------
    FileNotFoundError
        If no ``config.json`` is present in *model_dir*.
    """
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf‑8") as fh:
        cfg: Dict[str, Any] = json.load(fh)

    channels = cfg.get("model_channels") or cfg.get("args", {}).get("channels")
    if channels is None:
        # Ultimate fallback – infer from input shape if present.
        channels = (cfg.get("input_shape", [None, None, None, 1])[-1]) or 1
        logger.warning(
            "Channels not explicitly declared. Falling back to %d (model: %s)",
            channels,
            model_dir.name,
        )
    return int(channels)


def create_model_with_strategy(
    base_model_dir: str | os.PathLike[str],
    model_name: str,
    strategy: tf.distribute.Strategy,
) -> Tuple[tf.keras.Model, int]:
    """Load a model inside *strategy* scope.

    Parameters
    ----------
    base_model_dir : str or pathlib.Path
        Root directory that contains sub‑folders for each model.
    model_name : str
        Folder name of the model to load.
    strategy : tf.distribute.Strategy
        Distribution strategy (e.g. ``MirroredStrategy``) used to place the
        variables and computations on the desired devices.

    Returns
    -------
    (tf.keras.Model, int)
        Tuple of the loaded model instance and its expected channel count.

    Raises
    ------
    FileNotFoundError
        If the provided *model_name* directory does not exist.
    """
    model_dir = Path(base_model_dir) / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    with strategy.scope():
        model = load_model_from_directory(model_dir, compile=False)
    channels = get_model_channels(model_dir)

    logger.info(
        "Model '%s' loaded from '%s' (channels=%d)", model.name, model_dir, channels
    )
    return model, channels


def _output_prediction_error(
    group_id_tensor: tf.Tensor,
    identifier_tensor: tf.Tensor,
    unique_key_tensor: tf.Tensor,
    error_queue: mp.Queue,
    error_text: str,
) -> None:
    """Log a prediction‑time error for a single sample.

    Some batches may partially fail (e.g. OOM on a slice). For each image that
    could not be processed we emit an explicit log entry to aid debugging and
    potential re‑processing.

    Parameters
    ----------
    group_id_tensor : tf.Tensor
        Tensor containing the *group ID*.
    identifier_tensor : tf.Tensor
        Tensor containing the *image identifier*.
    unique_key_tensor : tf.Tensor
        Tensor containing the unique request key for SSE routing.
    error_queue : mp.Queue
        The queue to which error messages are sent.
    error_text : str
        Human‑readable error description.
    """
    try:
        group_id = group_id_tensor.numpy().decode("utf‑8", "ignore")
        identifier = identifier_tensor.numpy().decode("utf‑8", "ignore")
        unique_key = unique_key_tensor.numpy().decode("utf‑8", "ignore")
        if not identifier and not group_id:  # padding row
            return
        logger.error(
            "Prediction error for %s/%s – %s", group_id, identifier, error_text
        )
        error_payload = {
            "group_id": group_id,
            "identifier": identifier,
            "error": "PredictionFailed",
            "detail": error_text,
        }
        error_queue.put((error_payload, unique_key), timeout=5.0)
    except Exception as exc:
        logger.critical("Failed to send prediction error to client: %s", exc)


def _safe_predict(
    model: tf.keras.Model,
    images: tf.Tensor,
    groups: tf.Tensor,
    identifiers: tf.Tensor,
    batch_id: str,
    unique_keys: tf.Tensor,
    error_queue: mp.Queue,
    *,
    min_chunk: int = 1,
) -> Optional[np.ndarray]:
    """Run ``model.predict_on_batch`` with OOM resilience.

    The batch is recursively bisected upon encountering
    :class:`tf.errors.ResourceExhaustedError` until the chunk size reaches
    *min_chunk*.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model used for inference.
    images : tf.Tensor
        4‑D tensor containing the input images.
    groups : tf.Tensor
        Tensor matching *images* on axis‑0 with group IDs.
    identifiers : tf.Tensor
        Tensor matching *images* on axis‑0 with unique image IDs.
    batch_id : str
        Batch UUID – used only for logging.
    unique_keys : tf.Tensor
        Tensor with unique request keys for SSE routing.
    error_queue : mp.Queue
        Queue to send error messages to.
    min_chunk : int, default=1
        Hard lower‑bound on how far we are willing to split.

    Returns
    -------
    np.ndarray | None
        Concatenated predictions or *None* if every chunk failed.
    """
    stack: List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, str]] = [
        (images, groups, identifiers, unique_keys, batch_id)
    ]
    predictions: List[np.ndarray] = []

    while stack:
        imgs, grp_t, id_t, ukeys_t, bid = stack.pop()
        try:
            predictions.append(model.predict_on_batch(imgs))
        except tf.errors.ResourceExhaustedError as oom:
            n_samples = int(imgs.shape[0])
            if n_samples <= min_chunk:
                _output_prediction_error(
                    grp_t[0], id_t[0], ukeys_t[0], error_queue, f"OOM: {oom}"
                )
                continue
            mid = n_samples // 2
            stack.append(
                (imgs[mid:], grp_t[mid:], id_t[mid:], ukeys_t[mid:], f"{bid}-B")
            )
            stack.append(
                (imgs[:mid], grp_t[:mid], id_t[:mid], ukeys_t[:mid], f"{bid}-A")
            )
        except Exception as exc:
            logger.error("Batch %s failed: %s", bid, exc, exc_info=True)
            for g, i, uk in zip(grp_t, id_t, ukeys_t):
                _output_prediction_error(g, i, uk, error_queue, f"Predict error: {exc}")

    if not predictions:
        return None
    return np.concatenate(predictions, axis=0)


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
    """Forward predictions and auxiliary tensors to the *decoder* queue."""
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
    error_queue: Any,
    stop_event: Any,
    strategy: tf.distribute.Strategy,
):
    """Continuous prediction loop that exits when *stop_event* is set."""
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
                unique_keys,
                error_queue,
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

        # Were we asked to swap models while iterating?
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
    mp_final_results_queue: mp.Queue,
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
    mp_final_results_queue : mp.Queue
        Queue to send error messages directly to the final consumer.
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
        mp_final_results_queue,
        stop_event,
        strategy,
    )
