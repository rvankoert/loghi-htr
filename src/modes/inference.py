# Imports

# > Standard library
import argparse
import logging
from typing import List

# > Local dependencies
from data.generator import DataGenerator
from data.loader import DataLoader
from utils.utils import Utils, decode_batch_predictions, normalize_confidence
from model.management import get_prediction_model

# > Third-party dependencies
import tensorflow as tf


def perform_inference(args: argparse.Namespace, model: tf.keras.Model,
                      inference_dataset: DataGenerator, char_list: List[str],
                      loader: DataLoader) -> None:
    """
    Performs inference on a given dataset using a specified model and writes
    the results to a file.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing arguments for the inference process. This
        includes settings like the path for the results file, whether to use
        masking, the batch size, and parameters for decoding predictions.
    model : tf.keras.Model
        The Keras model to be used for inference.
    inference_dataset : DataGenerator
        The dataset on which inference is to be performed.
    char_list : List[str]
        A list of characters used in the model, for decoding predictions.
    loader : DataLoader
        A data loader object used for retrieving additional information needed
        during inference (e.g., filenames).

    Notes
    -----
    This function processes each batch from the inference dataset, predicts
    using the model, decodes these predictions into readable text, and writes
    them along with their confidence scores to a specified results file. It
    also handles the normalization of confidence scores and formatting of
    results.
    """

    utils_object = Utils(char_list, args.use_mask)
    prediction_model = get_prediction_model(model)

    with open(args.results_file, "w") as results_file:
        for batch_no, batch in enumerate(inference_dataset):
            # Get the predictions
            predictions = prediction_model.predict(batch[0], verbose=0)
            y_pred = decode_batch_predictions(
                predictions, utils_object, args.greedy,
                args.beam_width, args.num_oov_indices)[0]

            # Print the predictions and process the CER
            for index, (confidence, prediction) in enumerate(y_pred):
                # Normalize the confidence before processing because it was
                # determined on the original prediction
                normalized_confidence = normalize_confidence(
                    confidence, prediction)

                # Remove the special characters from the prediction
                prediction = prediction.strip().replace('', '')

                # Format the filename
                filename = loader.get_item(
                    'inference', (batch_no * args.batch_size) + index)

                # Write the results to the results file
                result_str = f"{filename}\t{normalized_confidence}\t" \
                    f"{prediction}"
                logging.info(result_str)
                results_file.write(result_str+"\n")

                # Flush the results file
                results_file.flush()