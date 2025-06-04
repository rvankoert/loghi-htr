# Imports

# > Standard library
import json
import logging
import multiprocessing
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

# > Third-party dependencies
import httpx
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
        raise FileNotFoundError(error_msg)

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in config file: %s", e)
        raise

    def search_key(data: Dict, key: str) -> Optional:
        """
        Recursively search for a key in a nested dictionary.

        Parameters
        ----------
        data : Dict
            Dictionary to search within.
        key : str
            Key to search for.

        Returns
        -------
        Optional
            Value associated with the key if found, else None.
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
        for key_bytes in whitelist:
            try:
                key = key_bytes.numpy().decode("utf-8")
            except UnicodeDecodeError as e:
                logging.warning("Error decoding whitelist key: %s", e)
                key = "INVALID_KEY"

            value = search_key(config, key)
            if value is None:
                values[key] = "NOT_FOUND"
                if key not in warned_keys:
                    logging.warning(
                        "Key '%s' not found in config. Recording as 'NOT_FOUND'", key
                    )
                    warned_keys.add(key)
            else:
                values[key] = value

        metadata_json = json.dumps(values)
        metadata_list.append(metadata_json)

    return metadata_list


def save_prediction_outputs(
    prediction_data: List[Tuple[float, str]],
    group_ids: List[bytes],
    image_ids: List[bytes],
    base_output_path: str,
    image_metadata: List[str],
    callback_url: str = None,
    temp_dir: Optional[str] = None,
    bidirectional: bool = False,
) -> List[str]:
    """
    Save decoded predictions to output files atomically.

    Parameters
    ----------
    prediction_data : List[Tuple[float, str]]
        List of tuples containing confidence scores and predicted texts.
    group_ids : List[bytes]
        List of group IDs for each image.
    image_ids : List[bytes]
        List of unique identifiers for each image.
    base_output_path : str
        Directory where prediction outputs should be saved.
    image_metadata : List[str]
        List of metadata JSON strings for each image.
    callback_urls: str
        Callback URL
    temp_dir : Optional[str], default=None
        Directory to use for temporary files. If None, a subdirectory of
        `base_output_path` is used.
    bidirectional : bool, default=False
        Whether to apply bidirectional text processing.

    Returns
    -------
    List[str]
        List of output texts for each image.

    Raises
    ------
    IOError
        If writing to any output file fails.
    """
    output_texts = []

    # Use provided temp_dir or create a default one
    if temp_dir is None:
        temp_dir = os.path.join(base_output_path, ".temp_prediction_outputs")

    os.makedirs(temp_dir, exist_ok=True)
    logging.debug("Using temporary directory: %s", temp_dir)

    for prediction, group_id, image_id, metadata in zip(
        prediction_data, group_ids, image_ids, image_metadata
    ):
        try:
            group_id = group_id.numpy().decode("utf-8")
            image_id = image_id.numpy().decode("utf-8")
        except UnicodeDecodeError as e:
            logging.error(traceback.format_exc())
            logging.error("Error decoding group_id or image_id: %s", e)
            continue  # Skip this entry

        confidence, predicted_text = prediction
        if bidirectional:
            predicted_text = get_display(predicted_text)
        output_text = f"{image_id}\t{metadata}\t{confidence}\t{predicted_text}"
        output_texts.append(output_text)

        try:
            payload = {
                "group_id": group_id,
                "identifier": image_id,
                "status": "predict OK",
                "result": output_text,
            }
            attempt_callback(callback_url, payload)
            logging.debug("Made callback to %s", callback_url)

        except Exception as e:
            logging.error("Failed to make callback. Error: %s", e)
            raise e

    return output_texts


def batch_decoding_worker(
    predicted_queue: multiprocessing.Queue,
    base_model_dir: str,
    model_path: str,
    output_path: str,
    stop_event: multiprocessing.Event,
    callback_url: str,
) -> None:
    """
    Worker function for processing and decoding batches of predictions.

    Parameters
    ----------
    predicted_queue : multiprocessing.Queue
        Queue containing predicted texts and associated metadata.
    base_model_dir : str
        Base directory where models are stored.
    model_path : str
        Relative path to the specific model directory within `base_model_dir`.
    output_path : str
        Directory where decoded prediction outputs should be saved.
    stop_event : multiprocessing.Event
        Event to signal the process to stop.
    status_queue : mp.Queue
        Queue for sending status updates back to the main process.

    Raises
    ------
    Exception
        Propagates any exception that occurs during the decoding process.
    """
    logging.info("Batch decoding process started")

    # Disable GPU for decoding
    tf.config.set_visible_devices([], "GPU")

    # Initialize tokenizer
    current_model_name = model_path
    tokenizer = create_tokenizer(os.path.join(base_model_dir, model_path))

    total_outputs = 0

    try:
        while not stop_event.is_set():
            try:
                # Attempt to retrieve data from the queue with a timeout
                data = predicted_queue.get(timeout=0.1)
                (
                    encoded_predictions,
                    batch_groups,
                    batch_identifiers,
                    model,
                    batch_id,
                    batch_whitelists,
                ) = data
            except multiprocessing.queues.Empty:
                continue  # No data available, continue checking

            # Re-initialize tokenizer if model has changed
            if model != current_model_name and model is not None:
                tokenizer = create_tokenizer(os.path.join(base_model_dir, model))
                logging.info("Tokenizer re-initialized for model: %s", model)
                current_model_name = model

            # Decode predictions
            decoded_predictions = batch_decode(encoded_predictions, tokenizer)

            # Fetch metadata based on whitelist keys
            batch_metadata = fetch_metadata(batch_whitelists, base_model_dir, model)

            # Save decoded predictions to output files
            outputted_predictions = save_prediction_outputs(
                decoded_predictions,
                batch_groups,
                batch_identifiers,
                output_path,
                batch_metadata,
                temp_dir=None,
                callback_url=callback_url,
                bidirectional=False,
            )
            total_outputs += len(outputted_predictions)

            # Log each outputted prediction
            for output in outputted_predictions:
                logging.debug("Outputted prediction: %s", output)

            logging.info(
                "Decoded and outputted batch %s (%d items)",
                batch_id,
                len(decoded_predictions),
            )
            logging.info("Total predictions completed: %d", total_outputs)

    except Exception as e:
        logging.error("Error in batch decoding process: %s", e)
        raise e

    logging.info("Batch decoding process stopped")
