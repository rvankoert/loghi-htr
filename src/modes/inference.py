# Imports

# > Standard library
import logging
from typing import List

# > Local dependencies
from data.generator import DataGenerator
from data.loader import DataLoader
from utils.decoding import decode_batch_predictions
from model.management import get_prediction_model
from setup.config import Config
from utils.text import Tokenizer

# > Third-party dependencies
import tensorflow as tf


def perform_inference(config: Config, model: tf.keras.Model,
                      inference_dataset: DataGenerator, charlist: List[str],
                      loader: DataLoader) -> None:
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
    inference_dataset : DataGenerator
        The dataset on which inference is to be performed.
    charlist : List[str]
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

    args = config.args

    tokenizer = Tokenizer(charlist, args.use_mask)
    prediction_model = get_prediction_model(model)

    with open(args.results_file, "w") as results_file:
        for batch_no, batch in enumerate(inference_dataset):
            # Get the predictions
            predictions = prediction_model.predict(batch[0], verbose=0)
            y_pred = decode_batch_predictions(predictions, tokenizer,
                                              args.greedy, args.beam_width)

            # Print the predictions and process the CER
            for index, (confidence, prediction) in enumerate(y_pred):
                # Remove the special characters from the prediction
                prediction = prediction.strip().replace('', '')

                # Format the filename
                filename = loader.get_item('inference',
                                           (batch_no * args.batch_size) + index
                                           )

                # Write the results to the results file
                result_str = f"{filename}\t{confidence}\t{prediction}"
                logging.info(result_str)
                results_file.write(result_str+"\n")

                # Flush the results file
                results_file.flush()
