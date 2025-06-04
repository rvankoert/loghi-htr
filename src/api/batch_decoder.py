# Imports

# > Standard library
import json
import logging
import multiprocessing
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple
from multiprocessing.queues import Full  # For type hinting

# > Third-party dependencies
import httpx  # Keep for attempt_callback
import numpy as np
import tensorflow as tf
from bidi.algorithm import get_display

# > Local imports
from callbacks import attempt_callback

# Add parent directory to path for local imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from utils.decoding import decode_batch_predictions  # noqa: E402
from utils.text import Tokenizer  # noqa: E402


def batch_decode(encoded_predictions: np.ndarray, tokenizer: Tokenizer) -> List[str]:
    """
    Decode a batch of encoded predictions into readable text.

    Parameters
    ----------
    encoded_predictions : np.ndarray
        Array of encoded predictions.
    tokenizer : Tokenizer
        Tokenizer utility for decoding predictions.

    Returns
    -------
    List[str]
        List of decoded prediction strings.
    """
    logging.debug("Starting batch decoding")
    decoded_predictions = decode_batch_predictions(encoded_predictions, tokenizer)
    logging.debug("Batch decoding completed")
    return decoded_predictions


def create_tokenizer(model_dir: str) -> Tokenizer:
    """
    Initialize a tokenizer utility for decoding predictions.

    Parameters
    ----------
    model_dir : str
        Directory path of the specific model.

    Returns
    -------
    Tokenizer
        Initialized tokenizer utility.

    Raises
    ------
    FileNotFoundError
        If the tokenizer file is not found in the model directory.
    Exception
        If an unexpected error occurs during tokenizer initialization.
    """
    tokenizer_file = None
    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    charlist_txt = os.path.join(model_dir, "charlist.txt")

    if os.path.exists(tokenizer_json):
        tokenizer_file = tokenizer_json
    elif os.path.exists(charlist_txt):
        tokenizer_file = charlist_txt
    else:
        error_msg = f"Tokenizer file not found in {model_dir}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        tokenizer = Tokenizer.load_from_file(tokenizer_file)
        logging.debug("Tokenizer initialized from %s", tokenizer_file)
    except FileNotFoundError as e:
        logging.error("Tokenizer file not found: %s", e)
        raise
    except Exception as e:
        logging.error("Error loading tokenizer: %s", e)
        raise

    return tokenizer


def fetch_metadata(
    whitelists: List[List[bytes]], base_model_dir: str, model_path: str
) -> List[str]:
    """
    Retrieve metadata based on whitelist keys from the model's configuration.

    Parameters
    ----------
    whitelists : List[List[bytes]]
        List of whitelist key lists for each prediction.
    base_model_dir : str
        Base directory where models are stored.
    model_path : str
        Relative path to the specific model directory within `base_model_dir`.

    Returns
    -------
    List[str]
        List of JSON strings containing metadata for each prediction.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    json.JSONDecodeError
        If the configuration file is not valid JSON.
    """
    config_path = os.path.join(base_model_dir, model_path, "config.json")

    if not os.path.exists(config_path):
        error_msg = f"Config file not found: {config_path}"
        logging.error(error_msg)
        # Return empty metadata if config not found, or handle as error
        return ["{}" for _ in whitelists]  # Or raise FileNotFoundError(error_msg)

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config_data = json.load(file)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in config file: %s. Using empty metadata.", e)
        return ["{}" for _ in whitelists]  # Or raise

    def search_key(data: Dict, key: str) -> Optional:
        """
        Recursively search for a key in a nested dictionary.
        """
        if key in data:
            return data[key]
        for sub_value in data.values():
            if isinstance(sub_value, dict):
                result = search_key(sub_value, key)
                if result is not None:
                    return result
        return None

    metadata_list = []
    warned_keys = set()

    for whitelist in whitelists:
        values = {}
        for key_bytes_tensor in whitelist:  # whitelist is now a list of tf.Tensor
            try:
                key = key_bytes_tensor.numpy().decode("utf-8")
                if not key:  # Skip empty strings that might come from padding
                    continue
            except UnicodeDecodeError as e:
                logging.warning("Error decoding whitelist key: %s", e)
                key = "INVALID_KEY"  # Or skip
            except (
                AttributeError
            ):  # If it's not a tensor (e.g. already a string if testing)
                key = key_bytes_tensor

            value = search_key(config_data, key)
            if value is None:
                values[key] = "NOT_FOUND"
                if key not in warned_keys:
                    logging.warning(
                        "Key '%s' not found in config. Recording as 'NOT_FOUND'", key
                    )
                    warned_keys.add(key)
            else:
                values[key] = value

        if not values and any(
            key.numpy().decode("utf-8", errors="ignore")
            for key in whitelist
            if hasattr(key, "numpy")
        ):
            # If whitelist had actual keys but none were found, log it.
            logging.debug(
                f"No whitelist keys found in config for one item. Whitelist: {[k.numpy().decode('utf-8', errors='ignore') for k in whitelist if hasattr(k, 'numpy')]}"
            )

        metadata_json = json.dumps(values)
        metadata_list.append(metadata_json)

    return metadata_list


def save_prediction_outputs(
    prediction_data: List[Tuple[float, str]],
    group_ids: List[bytes],  # These are tf.Tensor EagerTensor bytes
    image_ids: List[bytes],  # These are tf.Tensor EagerTensor bytes
    base_output_path: str,
    image_metadata: List[str],
    mp_final_results_queue: Optional[multiprocessing.Queue] = None,  # New parameter
    temp_dir: Optional[str] = None,  # Not currently used by this function
    bidirectional: bool = False,
) -> List[str]:
    """
    Save decoded predictions to output files atomically (original intent, now primarily formats and sends).
    Also sends results to mp_final_results_queue for SSE streaming.
    """
    output_texts_for_logging = []  # Renamed to avoid confusion
    logger = logging.getLogger(__name__)

    for prediction, group_id_tensor, image_id_tensor, metadata_str in zip(
        prediction_data, group_ids, image_ids, image_metadata
    ):
        try:
            group_id = (
                group_id_tensor.numpy().decode("utf-8")
                if hasattr(group_id_tensor, "numpy")
                else group_id_tensor
            )
            image_id = (
                image_id_tensor.numpy().decode("utf-8")
                if hasattr(image_id_tensor, "numpy")
                else image_id_tensor
            )

            if not image_id and not group_id:
                logger.debug("Skipping padded/empty entry in save_prediction_outputs.")
                continue
        except UnicodeDecodeError as e:
            logger.error(
                "Error decoding group_id or image_id in save_prediction_outputs: %s",
                e,
                exc_info=True,
            )
            continue

        confidence, predicted_text = prediction
        if bidirectional:
            predicted_text = get_display(predicted_text)

        formatted_confidence = (
            float(confidence)
            if isinstance(confidence, (np.float32, np.float64))
            else confidence
        )

        # Log the output locally (optional)
        log_output = (
            f"{image_id}\t{metadata_str}\t{formatted_confidence}\t{predicted_text}"
        )
        output_texts_for_logging.append(log_output)
        # logger.debug(f"Formatted prediction for logging/debug: {log_output}")

        # Push to mp_final_results_queue for SSE
        if mp_final_results_queue:
            try:
                # Ensure metadata_str is valid JSON before trying to load it
                try:
                    parsed_metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in metadata_str for {image_id}: '{metadata_str}'. Sending as raw string."
                    )
                    parsed_metadata = {"raw_metadata": metadata_str}

                result_for_sse = {
                    "group_id": group_id,
                    "identifier": image_id,
                    "text": predicted_text,
                    "confidence": formatted_confidence,
                    "metadata": parsed_metadata,
                }
                mp_final_results_queue.put(result_for_sse, timeout=5.0)
                logger.debug(
                    f"Pushed result for {image_id} to final results queue for SSE."
                )
            except Full:
                logger.error(
                    f"MP Final Results Queue is full. Dropping SSE result for {image_id}."
                )
            except Exception as e:
                logger.error(
                    f"Error pushing result for {image_id} to MP Final Results Queue: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "mp_final_results_queue not provided to save_prediction_outputs. SSE will not receive results."
            )

    return output_texts_for_logging


def batch_decoding_worker(
    predicted_batches_queue: multiprocessing.Queue,  # Input: batches of (encoded_preds, groups, ids, model, batch_id, whitelists)
    mp_final_results_queue: multiprocessing.Queue,  # Output: individual {group_id, id, text, confidence, metadata} dicts
    base_model_dir: str,
    model_path: str,  # Initial model path
    output_path: str,  # Base output for any files, not primarily used now for results
    stop_event: multiprocessing.Event,
) -> None:
    """
    Worker function for processing and decoding batches of predictions.
    """
    logger = logging.getLogger(__name__)
    logger.info("Batch decoding process started")

    tf.config.set_visible_devices([], "GPU")  # Keep CPU for decoding

    current_model_name = model_path
    try:
        tokenizer = create_tokenizer(os.path.join(base_model_dir, model_path))
    except Exception as e:
        logger.critical(
            f"Decoder: Failed to initialize tokenizer for {model_path} on startup: {e}. Worker exiting.",
            exc_info=True,
        )
        return

    total_outputs_processed = 0
    try:
        while not stop_event.is_set():
            try:
                data = predicted_batches_queue.get(timeout=0.1)
                if data is None:
                    logger.info("Decoding worker received sentinel. Exiting.")
                    break
                (
                    encoded_predictions,
                    batch_groups,
                    batch_identifiers,
                    batch_model_path,
                    batch_id,
                    batch_whitelists,
                ) = data
            except multiprocessing.queues.Empty:
                continue
            except Exception as e:
                logger.error(
                    f"Decoder: Error getting data from predicted_batches_queue: {e}",
                    exc_info=True,
                )
                continue

            if batch_model_path != current_model_name and batch_model_path is not None:
                try:
                    tokenizer = create_tokenizer(
                        os.path.join(base_model_dir, batch_model_path)
                    )
                    logger.info(
                        "Decoder: Tokenizer re-initialized for model: %s",
                        batch_model_path,
                    )
                    current_model_name = batch_model_path
                except Exception as e:
                    logger.error(
                        f"Decoder: Failed to re-initialize tokenizer for {batch_model_path}: {e}. Skipping batch {batch_id}.",
                        exc_info=True,
                    )
                    continue

            decoded_predictions_with_conf = batch_decode(encoded_predictions, tokenizer)
            batch_metadata_json_strings = fetch_metadata(
                batch_whitelists, base_model_dir, current_model_name
            )

            outputted_for_log = save_prediction_outputs(
                decoded_predictions_with_conf,
                batch_groups,
                batch_identifiers,
                output_path,
                batch_metadata_json_strings,
                mp_final_results_queue=mp_final_results_queue,  # Pass the queue for SSE
                bidirectional=False,
            )

            actual_outputted_count = len(
                outputted_for_log
            )  # Number of non-padding items
            total_outputs_processed += actual_outputted_count

            if actual_outputted_count > 0:
                logger.info(
                    "Decoded and processed batch %s for SSE (%d actual items)",
                    batch_id,
                    actual_outputted_count,
                )
            logger.debug(
                "Total individual predictions processed by decoder: %d",
                total_outputs_processed,
            )

    except Exception as e:
        logger.critical(f"Critical error in batch decoding process: {e}", exc_info=True)
    finally:
        logger.info(
            "Batch decoding process stopped. Total items processed: %d",
            total_outputs_processed,
        )
