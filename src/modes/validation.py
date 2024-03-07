# Imports

# > Standard library
from collections import defaultdict
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.creator import DataCreator
from model.management import get_prediction_model
from setup.config import Config
from utils.calculate import calc_95_confidence_interval, \
    calculate_edit_distances, update_statistics, increment_counters
from utils.decoding import decode_batch_predictions
from utils.print import print_predictions, display_statistics
from utils.text import preprocess_text, Tokenizer, normalize_text
from utils.wbs import setup_word_beam_search, handle_wbs_results


def process_batch(batch: Tuple[tf.Tensor, tf.Tensor],
                  prediction_model: tf.keras.Model,
                  tokenizer: Tokenizer,
                  config: Config,
                  wbs: Optional[Any],
                  loader: DataCreator,
                  batch_no: int,
                  chars: List[str]) -> Dict[str, int]:
    """
    Processes a batch of data by predicting, calculating Character Error Rate
    (CER), and handling Word Beam Search (WBS) if enabled.

    Parameters
    ----------
    batch : Tuple[tf.Tensor, tf.Tensor]
        A tuple containing the input data (X) and true labels (y_true) for the
        batch.
    prediction_model : tf.keras.Model
        The prediction model derived from the main model for inference.
    tokenizer : Tokenizer
        A tokenizer object for converting between characters and integers.
    config : Config
        A Config object containing the configuration for processing the batch,
        like batch size and settings for WBS.
    wbs : Optional[Any]
        An optional Word Beam Search object for advanced decoding, if
        applicable.
    loader : DataLoader
        A data loader object for additional operations like normalization.
    batch_no : int
        The number of the current batch being processed.
    chars : List[str]
        A list of characters used in the model.

    Returns
    -------
    Dict[str, int]
        A dictionary containing various counts and statistics computed during
        the batch processing, such as CER.
    """

    X, y_true = batch

    # Get the predictions
    predictions = prediction_model.predict_on_batch(X)
    y_pred = decode_batch_predictions(predictions, tokenizer, config["greedy"],
                                      config["beam_width"])

    # Transpose the predictions for WordBeamSearch
    if wbs:
        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
        char_str = handle_wbs_results(predsbeam, wbs, chars)
    else:
        char_str = None

    # Initialize the batch counter
    batch_counter = defaultdict(int)

    # Print the predictions and process the CER
    for index, (confidence, prediction) in enumerate(y_pred):
        # Preprocess the text for CER calculation
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(y_true[index])
        normalized_original = None if not config["normalization_file"] else \
            normalize_text(original_text, config["normalization_file"])

        # Calculate edit distances here so we can use them for printing the
        # predictions
        distances = calculate_edit_distances(prediction, original_text)
        normalized_distances = None if not config["normalization_file"] else \
            calculate_edit_distances(prediction, normalized_original)
        wbs_distances = None if not wbs else \
            calculate_edit_distances(char_str[index], original_text)

        # Print the predictions if there are any errors
        if do_print := distances[0] > 0:
            filename = loader.get_filename('validation',
                                           (batch_no * config["batch_size"])
                                           + index)
            wbs_str = char_str[index] if wbs else None

            print_predictions(filename, original_text, prediction,
                              normalized_original, wbs_str)
            logging.info("Confidence = %.4f", confidence)
            logging.info("")

        # Wrap the distances and originals in dictionaries
        distances_dict = {"distances": distances,
                          "Normalized distances": normalized_distances,
                          "WBS distances": wbs_distances}
        original_dict = {"original": original_text,
                         "Normalized original": normalized_original,
                         "WBS original": original_text}

        # Update the batch counter
        batch_counter = increment_counters(distances_dict,
                                           original_dict,
                                           batch_counter,
                                           do_print)

    return batch_counter


def perform_validation(config: Config,
                       model: tf.keras.Model,
                       charlist: List[str],
                       dataloader: DataCreator) -> None:
    """
    Performs validation on a dataset using a given model and calculates various
    metrics like Character Error Rate (CER).

    Parameters
    ----------
    config : Config
        A Config object containing the configuration for the validation
        process such as mask usage and file paths.
    model : tf.keras.Model
        The Keras model to be validated.
    charlist : List[str]
        A list of characters used in the model.
    dataloader : DataLoader
        A data loader object for additional operations like normalization and
        Word Beam Search setup.

    Notes
    -----
    The function processes each batch in the validation dataset, calculates
    CER, and optionally processes Word Beam Search (WBS) results if enabled.
    It also handles the display and logging of statistical information
    throughout the validation process.
    """

    logging.info("Performing validation...")

    tokenizer = dataloader.tokenizer
    validation_dataset = dataloader.datasets['validation']

    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(config, charlist) \
        if config["corpus_file"] else None

    # Initialize variables for CER calculation
    n_items = 0
    total_counter = defaultdict(int)

    # Process each batch in the validation dataset
    for batch_no, batch in enumerate(validation_dataset):
        X = batch[0]
        y = [dataloader.get_ground_truth('validation', i)
             for i in range(batch_no * config["batch_size"],
                            batch_no * config["batch_size"] + len(X))]

        # Logic for processing each batch, calculating CER, etc.
        batch_counter = process_batch((X, y), prediction_model, tokenizer,
                                      config, wbs, dataloader, batch_no,
                                      charlist)

        # Update totals with batch information
        for key, value in batch_counter.items():
            total_counter[key] += value

        # Calculate the CER
        metrics, batch_stats, total_stats = update_statistics(batch_counter,
                                                              total_counter)

        # Add the number of items to the total tally
        n_items += len(batch[1])
        metrics.append('Items')
        batch_stats.append(len(batch[1]))
        total_stats.append(n_items)

        # Print batch info
        display_statistics(batch_stats, total_stats, metrics)
        logging.info("")

    # Print the final validation statistics
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final validation statistics")
    logging.info("---------------------------")

    # Calculate the CER confidence intervals on all metrics except Items
    intervals = [calc_95_confidence_interval(cer_metric, n_items)
                 for cer_metric in total_stats[:-1]]

    # Print the final statistics
    for metric, total_value, interval in zip(metrics[:-1], total_stats[:-1],
                                             intervals):
        logging.info("%s = %.4f +/- %.4f", metric, total_value, interval)

    logging.info("Items = %s", total_stats[-1])
    logging.info("")

    # Output the validation statistics to a csv file
    with open(os.path.join(config["output"], 'validation.csv'),
              'w', encoding="utf-8") as f:
        header = "cer,cer_lower,cer_simple"
        if config["normalization_file"]:
            header += ",normalized_cer,normalized_cer_lower," \
                "normalized_cer_simple"
        if wbs:
            header += ",wbs_cer,wbs_cer_lower,wbs_cer_simple"

        f.write(header + "\n")
        results = ",".join([str(total_stats[i])
                           for i in range(len(metrics)-1)])
        f.write(results + "\n")
