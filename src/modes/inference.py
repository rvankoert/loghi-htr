# Imports

# > Standard library
import logging
from typing import List

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.decoding import decode_batch_predictions
from utils.text import Tokenizer


def perform_inference(config: Config,
                      model: tf.keras.Model,
                      inference_dataset: tf.data.Dataset,
                      data_manager: DataManager) -> None:
    """
    Performs inference on a given dataset using a specified model and writes
    the results to a file.

    Parameters
    ----------
    config : Config
        A Config object containing arguments for the inference process. This
        includes settings like the path for the results file, whether to use
        masking, the batch size, and parameters for decoding predictions.
    model : tf.keras.Model
        The Keras model to be used for inference.
    inference_dataset : tf.data.Dataset
        The dataset on which inference is to be performed.
    data_manager : DataManager
        A data manager object used for retrieving additional information needed
        during inference (e.g., filenames).

    Notes
    -----
    This function processes each batch from the inference dataset, predicts
    using the model, decodes these predictions into readable text, and writes
    them along with their confidence scores to a specified results file. It
    also handles the normalization of confidence scores and formatting of
    results.
    """

    tokenizer = data_manager.tokenizer

    with open(config["results_file"], "w", encoding="utf-8") as results_file:
        for batch_no, batch in enumerate(inference_dataset):
            # Get the predictions
            predictions = model.predict_on_batch(batch[0])
            y_pred = decode_batch_predictions(predictions,
                                              tokenizer,
                                              config["greedy"],
                                              config["beam_width"])

            # Print the predictions and process the CER
            for index, (confidence, prediction) in enumerate(y_pred):
                # Remove the special characters from the prediction
                prediction = prediction.strip().replace('[MASK]', '')

                # Format the filename
                filename = data_manager.get_filename('inference',
                                                     (batch_no *
                                                      config["batch_size"])
                                                     + index)

                # Write the results to the results file
                result_str = f"{filename}\t{confidence}\t{prediction}"
                logging.info(result_str)
                results_file.write(result_str+"\n")

                # Flush the results file
                results_file.flush()
