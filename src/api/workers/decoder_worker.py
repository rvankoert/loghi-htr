# Imports

# > Standard library
import json
import logging
import multiprocessing as mp
import os
import sys
from multiprocessing.queues import (
    Empty as MPQueueEmptyException,
)
from multiprocessing.queues import (
    Full as MPQueueFullException,
)
from typing import Any, Dict, List, Optional, Tuple

# > Third-party dependencies
import numpy as np
import tensorflow as tf
from bidi.algorithm import get_display

# Correct sys.path modification for worker context
current_worker_file_dir = os.path.dirname(os.path.realpath(__file__))
api_dir = os.path.dirname(current_worker_file_dir)
src_dir = os.path.dirname(api_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# > Local imports
from utils.decoding import decode_batch_predictions
from utils.text import Tokenizer

logger = logging.getLogger(__name__)


def _batch_decode_predictions(
    encoded_predictions: np.ndarray, tokenizer: Tokenizer
) -> List[Tuple[float, str]]:
    """Decodes batch predictions, returns list of (confidence, text) tuples."""
    decoded = decode_batch_predictions(encoded_predictions, tokenizer)
    return decoded


def _create_tokenizer(model_dir: str) -> Tokenizer:
    """Initializes a tokenizer from model directory."""
    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    charlist_txt = os.path.join(model_dir, "charlist.txt")
    tokenizer_file = None

    if os.path.exists(tokenizer_json):
        tokenizer_file = tokenizer_json
    elif os.path.exists(charlist_txt):
        tokenizer_file = charlist_txt
    else:
        raise FileNotFoundError(f"Tokenizer file not found in {model_dir}")

    tokenizer = Tokenizer.load_from_file(tokenizer_file)
    logger.debug("Tokenizer initialized from %s", tokenizer_file)
    return tokenizer


def _fetch_metadata_for_batch(
    whitelists_tensor: tf.Tensor,
    base_model_dir: str,
    model_name: str,
) -> List[str]:
    """Retrieves metadata for a batch based on whitelists."""
    config_path = os.path.join(base_model_dir, model_name, "config.json")
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in config {config_path}: {e}. Using empty metadata."
            )

    def search_key_recursive(data_dict: Dict, key_str: str) -> Optional[Any]:
        if key_str in data_dict:
            return data_dict[key_str]
        for v in data_dict.values():
            if isinstance(v, dict):
                found = search_key_recursive(v, key_str)
                if found is not None:
                    return found
        return None

    batch_metadata_json = []
    warned_keys = set()

    # Handle if whitelists_tensor is RaggedTensor or regular Tensor
    # Convert to list of lists of strings
    processed_whitelists = []
    if isinstance(whitelists_tensor, tf.RaggedTensor):
        for i in range(whitelists_tensor.nrows()):
            row = whitelists_tensor.row(i)
            processed_whitelists.append(
                [key.numpy().decode("utf-8", "ignore") for key in row if key.numpy()]
            )
    else:  # Assuming tf.Tensor
        for i in range(tf.shape(whitelists_tensor)[0]):  # Iterate over batch dimension
            item_whitelist_tensor = whitelists_tensor[
                i
            ]  # This is a 1D tensor of strings for one item
            # Filter out padding strings (empty strings)
            processed_whitelists.append(
                [
                    key.numpy().decode("utf-8", "ignore")
                    for key in item_whitelist_tensor
                    if key.numpy()
                ]
            )

    for item_whitelist_keys in processed_whitelists:  # item_whitelist_keys is List[str]
        item_metadata = {}
        for key_str in item_whitelist_keys:
            if not key_str:
                continue  # Skip empty keys from padding
            value = search_key_recursive(config_data, key_str)
            if value is None:
                item_metadata[key_str] = "NOT_FOUND"
                if key_str not in warned_keys:
                    logger.warning(
                        f"Metadata key '{key_str}' not found in model '{model_name}' config. Marked 'NOT_FOUND'."
                    )
                    warned_keys.add(key_str)
            else:
                item_metadata[key_str] = value
        batch_metadata_json.append(json.dumps(item_metadata))
    return batch_metadata_json


def _format_and_send_decoded_results(
    predictions_with_conf: List[Tuple[float, str]],
    group_ids_tensor: tf.Tensor,
    image_ids_tensor: tf.Tensor,
    batch_metadata_json: List[str],
    unique_keys_tensor: tf.Tensor,
    mp_final_results_queue: mp.Queue,
    bidirectional_text: bool = False,
) -> int:  # Returns count of successfully sent items
    """Formats decoded predictions and sends them to the final results queue for SSE."""
    sent_count = 0
    for i, (confidence, text) in enumerate(predictions_with_conf):
        try:
            # Tensors contain bytes, need to decode
            group_id = group_ids_tensor[i].numpy().decode("utf-8", "ignore")
            image_id = image_ids_tensor[i].numpy().decode("utf-8", "ignore")
            metadata_json_str = batch_metadata_json[i]
            unique_key = unique_keys_tensor[i].numpy().decode("utf-8", "ignore")

            if not image_id and not group_id:  # Likely padding
                logger.debug(
                    "Skipping padded/empty entry in _format_and_send_decoded_results."
                )
                continue
        except Exception as e:
            logger.error(f"Error decoding IDs for item {i}: {e}", exc_info=True)
            continue

        if bidirectional_text:
            text = get_display(text)

        try:
            metadata_dict = json.loads(metadata_json_str)
        except json.JSONDecodeError:
            logger.warning(
                f"Invalid JSON metadata for {image_id}: '{metadata_json_str}'. Sending raw."
            )
            metadata_dict = {"raw_metadata": metadata_json_str}

        result_str = "\t".join(
            [image_id, json.dumps(metadata_dict), str(confidence), text]
        )

        result_for_sse = {
            "group_id": group_id,
            "identifier": image_id,
            "result": result_str,
        }

        try:
            mp_final_results_queue.put((result_for_sse, unique_key), timeout=5.0)
            sent_count += 1
            logger.debug(f"Pushed result for {image_id} to final results queue.")
        except MPQueueFullException:
            logger.error(
                f"MP Final Results Queue full. Dropping result for {image_id}."
            )
        except Exception as e:
            logger.error(
                f"Error pushing result for {image_id} to queue: {e}", exc_info=True
            )

    return sent_count


def decoder_process_entrypoint(
    mp_predicted_batches_queue: mp.Queue,
    mp_final_results_queue: mp.Queue,
    config: Dict[str, Any],
    stop_event: mp.Event,
):
    """Main function for the batch decoding worker process."""
    worker_config = {
        "base_model_dir": config["base_model_dir"],
        "initial_model_name": config["model_name"],
    }
    logger.info(f"Decoder worker starting with config: {worker_config}")

    # Decoder runs on CPU
    tf.config.set_visible_devices([], "GPU")

    current_model_name = worker_config["initial_model_name"]
    try:
        tokenizer = _create_tokenizer(
            os.path.join(worker_config["base_model_dir"], current_model_name)
        )
    except Exception as e:
        logger.critical(
            f"Decoder: Failed to load tokenizer for {current_model_name} on startup: {e}. Worker exiting.",
            exc_info=True,
        )
        return

    total_items_processed = 0
    try:
        while not stop_event.is_set():
            try:
                # Data: (encoded_preds_np, groups_tensor, ids_tensor, model_name_used, batch_uuid, whitelists_tensor)
                data_from_predictor = mp_predicted_batches_queue.get(timeout=0.1)
                if data_from_predictor is None:  # Sentinel
                    logger.info("Decoder worker received sentinel. Exiting.")
                    break

                (
                    encoded_preds_np,
                    groups_tensor,
                    ids_tensor,
                    batch_model_name,  # Model name used by predictor for this batch
                    batch_uuid,
                    whitelists_tensor,  # This is a tf.Tensor or tf.RaggedTensor
                    unique_keys_tensor,
                ) = data_from_predictor

            except MPQueueEmptyException:
                continue
            except ValueError:  # If tuple unpacking fails due to wrong number of items (e.g. old data in queue)
                logger.error(
                    "Decoder: Error unpacking data from queue, likely due to data format mismatch. Discarding item.",
                    exc_info=True,
                )
                continue
            except Exception as e:
                logger.error(
                    f"Decoder: Error getting data from predicted_batches_queue: {e}",
                    exc_info=True,
                )
                continue

            if batch_model_name != current_model_name:
                logger.info(
                    f"Decoder: Model changed from '{current_model_name}' to '{batch_model_name}'. Reloading tokenizer."
                )
                try:
                    tokenizer = _create_tokenizer(
                        os.path.join(worker_config["base_model_dir"], batch_model_name)
                    )
                    current_model_name = batch_model_name
                except Exception as e:
                    logger.error(
                        f"Decoder: Failed to re-init tokenizer for {batch_model_name}: {e}. Skipping batch {batch_uuid}.",
                        exc_info=True,
                    )
                    continue

            logger.debug(
                f"Decoding batch {batch_uuid} (size {len(encoded_preds_np)}) with model {current_model_name}"
            )
            decoded_preds_with_conf = _batch_decode_predictions(
                encoded_preds_np, tokenizer
            )

            batch_metadata_list_json = _fetch_metadata_for_batch(
                whitelists_tensor, worker_config["base_model_dir"], current_model_name
            )

            sent_count = _format_and_send_decoded_results(
                decoded_preds_with_conf,
                groups_tensor,
                ids_tensor,
                batch_metadata_list_json,
                unique_keys_tensor,
                mp_final_results_queue,
            )

            if sent_count > 0:
                total_items_processed += sent_count
                logger.info(
                    f"Decoded and sent {sent_count} items for batch {batch_uuid} to SSE queue."
                )
            logger.debug(f"Total items processed by decoder: {total_items_processed}")

    except Exception as e:
        logger.critical(f"Critical error in decoder worker: {e}", exc_info=True)
    finally:
        logger.info(
            f"Decoder worker stopped. Total items processed for SSE: {total_items_processed}."
        )
