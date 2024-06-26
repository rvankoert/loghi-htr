# Imports

# > Standard library
import logging
import multiprocessing
import os
import sys
import shutil
from typing import List, Tuple
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
                          model_path: str,
                          output_path: str,
                          stop_event: multiprocessing.Event) -> None:
    """
    Worker function for batch decoding process.

    Parameters
    ----------
    predicted_queue: multiprocessing.Queue
        Queue containing predicted texts and other information.
    model_path: str
        Path to the model directory.
    output_path: str
        Base path where prediction outputs should be saved.
    """

    logging.info("Batch decoding process started")

    # Disable GPU
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Initialize utilities
    tokenizer = create_tokenizer(model_path)

    total_outputs = 0

    try:
        while not stop_event.is_set():
            try:
                encoded_predictions, batch_groups, batch_identifiers, model, \
                    batch_id, batch_metadata = predicted_queue.get(timeout=0.1)
            except multiprocessing.queues.Empty:
                continue

            # Re-initialize utilities if model has changed
            if model != model_path:
                tokenizer = create_tokenizer(model)
                logging.info("Utilities re-initialized for %s", model)
                model_path = model

            decoded_predictions = batch_decode(encoded_predictions, tokenizer)

            logging.debug("Outputting predictions...")
            outputted_predictions = output_predictions(decoded_predictions,
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
    charlist_path = f"{model_path}/charlist.txt"
    try:
        with open(charlist_path, encoding="utf-8") as file:
            charlist = [char for char in file.read() if char != '']
        tokenizer = Tokenizer(charlist, use_mask=True)
        logging.debug("Utilities initialized")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"charlist.txt not found at {model_path}") \
            from e
    except Exception as e:
        logging.error("Error loading utilities: %s", e)
        raise e

    return tokenizer


def output_predictions(predictions: List[Tuple[float, str]],
                       groups: List[str],
                       identifiers: List[str],
                       output_path: str,
                       batch_metadata: List[dict]) -> List[str]:
    """
    Generate output texts based on the predictions and save to files
    atomically.

    Parameters
    ----------
    predictions: List[Tuple[float, str]]
        List of tuples containing confidence and predicted text for each image.
    groups: List[str]
        List of group IDs for each image.
    identifiers: List[str]
        List of identifiers for each image.
    output_path: str
        Base path where prediction outputs should be saved.
    batch_metadata: List[dict]
        List of metadata dictionaries for each image.

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
    outputs = []
    for i, (confidence, pred_text) in enumerate(predictions):
        group_id = groups[i]
        identifier = identifiers[i]
        metadata = batch_metadata[i]
        text = f"{identifier}\t{metadata}\t{confidence}\t{pred_text}"
        outputs.append(text)

        # Output the text to a file atomically
        output_dir = os.path.join(output_path, group_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.debug("Created output directory: %s", output_dir)

        final_path = os.path.join(output_dir, identifier + ".txt")

        # Create a temporary file in the same directory
        with tempfile.NamedTemporaryFile(mode='w',
                                         dir=output_path,
                                         delete=False,
                                         encoding="utf-8") as temp_file:
            temp_file.write(text + "\n")
            temp_path = temp_file.name

        # Atomically replace the destination file
        try:
            shutil.move(temp_path, final_path)
            logging.debug(f"Atomically wrote file: {final_path}")
        except Exception as e:
            logging.error("Failed to atomically write file: %s. Error: %s",
                          final_path, e)
            # Clean up the temporary file if the move failed
            os.unlink(temp_path)
            raise

    return outputs
