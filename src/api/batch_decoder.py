# Imports

# > Standard library
import logging
import multiprocessing
import numpy as np
import os
import sys
from typing import List, Tuple

# > Third-party dependencies
import tensorflow as tf

# > Local imports
# Add parent directory to path for imports
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from utils.utils import Utils, decode_batch_predictions, \
        normalize_confidence  # noqa: E402


def batch_decoding_worker(predicted_queue: multiprocessing.Queue,
                          model_path: str):
    """
    Worker function for batch decoding process.

    Parameters
    ----------
    predicted_queue: multiprocessing.Queue
        Queue containing predicted texts and other information.
    model_path: str
        Path to the model directory.
    """

    logging.info("Batch decoding process started")

    # Disable GPU
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Initialize utilities
    utils = create_utils(model_path)

    total_outputs = 0
    batch_num = 0

    try:
        while True:
            encoded_predictions, groups, identifiers, output_path, model = \
                predicted_queue.get()

            # Re-initialize utilities if model has changed
            if model != model_path:
                utils = create_utils(model)
                logging.info(f"Utilities re-initialized for {model}")
                model_path = model

            decoded_predictions = batch_decode(encoded_predictions, utils)

            logging.debug("Outputting predictions...")
            outputted_predictions = output_predictions(decoded_predictions,
                                                       groups,
                                                       identifiers,
                                                       output_path)
            total_outputs += len(outputted_predictions)

            for output in outputted_predictions:
                logging.debug(f"Outputted prediction: {output}")

            logging.info(f"Outputted batch {batch_num}")
            logging.info(f"Total predictions outputted: {total_outputs}")

            batch_num += 1

    except Exception as e:
        logging.error(f"Error in batch decoding process: {e}")
        raise e


def batch_decode(encoded_predictions: np.ndarray,
                 utils: Utils) -> List[str]:
    """
    Decode a batch of encoded predictions.

    Parameters
    ----------
    encoded_predictions: np.ndarray
        Array of encoded predictions.
    utils: Utils
        Utilities object containing character list and other information.

    Returns
    -------
    List[str]
        List of decoded predictions.
    """

    logging.debug("Decoding predictions...")
    decoded_predictions = decode_batch_predictions(
        encoded_predictions, utils)[0]
    logging.debug("Predictions decoded")

    return decoded_predictions


def create_utils(model_path: str) -> Utils:
    """
    Create a utilities object for decoding.

    Parameters
    ----------
    model_path: str
        Path to the model directory.

    Returns
    -------
    Utils
        Utilities object containing character list and other information.
    """

    # Load the character list
    charlist_path = f"{model_path}/charlist.txt"
    try:
        with open(charlist_path) as file:
            charlist = [char for char in file.read()]
        utils = Utils(charlist, use_mask=True)
        logging.debug("Utilities initialized")
    except FileNotFoundError:
        logging.error(f"charlist.txt not found at {model_path}. Exiting...")
        raise FileNotFoundError
    except Exception as e:
        logging.error(f"Error loading utilities: {e}")
        raise e

    return utils


def output_predictions(predictions: List[Tuple[float, str]],
                       groups: List[str],
                       identifiers: List[str],
                       output_path: str) -> List[str]:
    """
    Generate output texts based on the predictions and save to files.

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

    Returns
    -------
    List[str]
        List of output texts for each image.

    Side Effects
    ------------
    - Creates directories for groups if they don't exist.
    - Saves output texts to files within the respective group directories.
    - Logs messages regarding directory creation and saving.
    """

    outputs = []
    for i, (confidence, pred_text) in enumerate(predictions):
        group_id = groups[i]
        identifier = identifiers[i]
        confidence = normalize_confidence(confidence, pred_text)

        text = f"{identifier}\t{str(confidence)}\t{pred_text}"
        outputs.append(text)

        # Output the text to a file
        output_dir = os.path.join(output_path, group_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Created output directory: {output_dir}")
        with open(os.path.join(output_dir, identifier + ".txt"), "w") as f:
            f.write(text + "\n")

    return outputs
