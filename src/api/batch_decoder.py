# Imports

# > Standard library
import logging
import multiprocessing
import os
import sys
import shutil
from typing import List, Tuple, Dict
import tempfile

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# > Local imports
# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from utils.decoding import decode_batch_predictions  # noqa: E402
from utils.text import Tokenizer  # noqa: E402


def batch_decoding_worker(predicted_queue: multiprocessing.Queue,
                          base_model_dir: str,
                          model_path: str,
                          output_path: str,
                          stop_event: multiprocessing.Event) -> None:
    """
    Worker function for batch decoding process.

    Parameters
    ----------
    predicted_queue: multiprocessing.Queue
        Queue containing predicted texts and other information.
    base_model_dir: str
        Base path to the model directory.
    model_path: str
        Path to the model directory.
    output_path: str
        Base path where prediction outputs should be saved.
    stop_event: multiprocessing.Event
        Event to signal the process to stop.
    """

    logging.info("Batch decoding process started")

    # Disable GPU
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Initialize utilities
    model_name = model_path
    tokenizer = create_tokenizer(os.path.join(base_model_dir, model_path))

    total_outputs = 0

    try:
        while not stop_event.is_set():
            try:
                encoded_predictions, batch_groups, batch_identifiers, model, \
                    batch_id, batch_metadata = predicted_queue.get(timeout=0.1)
            except multiprocessing.queues.Empty:
                continue

            # Re-initialize utilities if model has changed
            if model != model_name:
                tokenizer = create_tokenizer(
                    os.path.join(base_model_dir, model))
                logging.info("Utilities re-initialized for %s", model)
                model_name = model

            decoded_predictions = batch_decode(encoded_predictions, tokenizer)

            logging.debug("Outputting predictions...")
            outputted_predictions = save_prediction_outputs(
                decoded_predictions,
                batch_groups,
                batch_identifiers,
                output_path,
                batch_metadata)
            total_outputs += len(outputted_predictions)

            for output in outputted_predictions:
                logging.debug("Outputted prediction: %s", output)

            logging.info("Decoded and outputted batch %s (%s items)",
                         batch_id, len(decoded_predictions))
            logging.info("Total predictions complete: %s", total_outputs)

    except Exception as e:
        logging.error("Error in batch decoding process: %s", e)
        raise e


def batch_decode(encoded_predictions: np.ndarray,
                 tokenizer: Tokenizer) -> List[str]:
    """
    Decode a batch of encoded predictions.

    Parameters
    ----------
    encoded_predictions: np.ndarray
        Array of encoded predictions.
    tokenizer: Tokenizer
        Utilities object containing character list and other information.

    Returns
    -------
    List[str]
        List of decoded predictions.
    """

    logging.debug("Decoding predictions...")
    decoded_predictions = decode_batch_predictions(encoded_predictions,
                                                   tokenizer)
    logging.debug("Predictions decoded")

    return decoded_predictions


def create_tokenizer(model_path: str) -> Tokenizer:
    """
    Create a utilities object for decoding.

    Parameters
    ----------
    model_path: str
        Path to the model directory.

    Returns
    -------
    Tokenizer
        Utilities object containing character list and other information.
    """

    # Load the character list
    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        charlist_path = os.path.join(model_path, "tokenizer.json")
    else:
        charlist_path = os.path.join(model_path, "charlist.txt")

    try:
        tokenizer = Tokenizer.load_from_file(charlist_path)
        logging.debug("Utilities initialized")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Tokenizer not found in {model_path}") \
            from e
    except Exception as e:
        logging.error("Error loading utilities: %s", e)
        raise e

    return tokenizer


def save_prediction_outputs(
    prediction_data: List[Tuple[float, str]],
    group_ids: List[str],
    image_ids: List[str],
    base_output_path: str,
    image_metadata: List[Dict],
    temp_dir: str = None
) -> List[str]:
    """
    Generate output texts based on predictions and save to files atomically.

    Parameters
    ----------
    prediction_data: List[Tuple[float, str]]
        List of tuples containing confidence and predicted text for each image.
    group_ids: List[str]
        List of group IDs for each image.
    image_ids: List[str]
        List of unique identifiers for each image.
    base_output_path: str
        Base path where prediction outputs should be saved.
    image_metadata: List[Dict]
        List of metadata dictionaries for each image.
    temp_dir: str, optional
        Path to use for temporary files. If None, a subdirectory of
        base_output_path is used.

    Returns
    -------
    List[str]
        List of output texts for each image.

    Side Effects
    ------------
    - Creates directories for groups if they don't exist.
    - Saves output texts to files within the respective group directories
      atomically.
    - Logs messages regarding directory creation and saving.
    """
    output_texts = []

    # If no temp_dir is provided, create one as a subdirectory of
    # base_output_path
    if temp_dir is None:
        temp_dir = os.path.join(base_output_path, '.temp_prediction_outputs')

    os.makedirs(temp_dir, exist_ok=True)
    logging.debug("Using temporary directory: %s", temp_dir)

    for prediction, group_id, image_id, metadata in zip(
        prediction_data, group_ids, image_ids, image_metadata
    ):
        confidence, predicted_text = prediction
        output_text = (
            f"{image_id}\t{metadata}\t{confidence}\t{predicted_text}")
        output_texts.append(output_text)

        group_output_dir = os.path.join(base_output_path, group_id)
        os.makedirs(group_output_dir, exist_ok=True)
        logging.debug("Ensured output directory exists: %s",
                      group_output_dir)

        output_file_path = os.path.join(
            group_output_dir, f"{image_id}.txt")

        try:
            write_file_atomically(output_text, output_file_path, temp_dir)
            logging.debug("Atomically wrote file: %s", output_file_path)
        except IOError as e:
            logging.error("Failed to write file %s. Error: %s",
                          output_file_path, e)
            raise

    return output_texts


def write_file_atomically(content: str, target_path: str, temp_dir: str) \
        -> None:
    """
    Write content to a file atomically, using a separate temporary directory.

    Parameters
    ----------
    content: str
        The content to write to the file.
    target_path: str
        The path where the file should be written.
    temp_dir: str
        Path to the temporary directory for intermediate files.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    temp_file_path = None
    try:
        # Create a temporary file in the provided temporary directory
        with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir,
                                         delete=False, encoding="utf-8") \
                as temp_file:
            temp_file.write(content + "\n")
            temp_file_path = temp_file.name

        # On POSIX systems, this is atomic. On Windows, it's the best we can do
        os.replace(temp_file_path, target_path)
    except IOError as e:
        if temp_file_path and os.path.exists(temp_file_path):
            # Clean up the temporary file if it exists
            os.unlink(temp_file_path)
        raise IOError(
            f"Failed to atomically write file: {target_path}. Error: {e}")
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path):
            # Clean up the temporary file if it exists
            os.unlink(temp_file_path)
        raise e  # Re-raise the exception after cleanup
