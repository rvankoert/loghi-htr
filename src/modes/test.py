# > Standard library
from collections import defaultdict
import logging
import os
from typing import Any, Dict, List, Optional

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.generator import DataGenerator
from data.loader import DataLoader
from model.management import get_prediction_model
from setup.config import Config
from utils.calculate import calculate_confidence_intervals, calculate_cers, \
    process_prediction_type
from utils.decoding import decode_batch_predictions
from utils.text import preprocess_text, Tokenizer
from utils.wbs import setup_word_beam_search, handle_wbs_results


def process_batch(y_true: tf.Tensor,
                  predictions: tf.Tensor,
                  tokenizer: Tokenizer,
                  config: Config,
                  wbs: Optional[Any],
                  loader: DataLoader,
                  batch_no: int,
                  chars: List[str]) -> Dict[str, int]:
    """
    Processes a batch of data by predicting, calculating Character Error Rate
    (CER), and handling Word Beam Search (WBS) if enabled.

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor containing the true labels for the batch.
    predictions : tf.Tensor
        A tensor containing the predictions for the batch.
    tokenizer : Tokenizer
        A tokenizer object for converting between characters and integers.
    config : Config
        A Config object containing arguments for processing the batch, like
        batch size and settings for WBS.
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

    # Get the predictions
    y_pred = decode_batch_predictions(predictions, tokenizer, config["greedy"],
                                      config["beam_width"])

    # Transpose the predictions for WordBeamSearch
    if wbs:
        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
        char_str = handle_wbs_results(predsbeam, wbs, chars)
    else:
        char_str = None

    # Get the original texts
    orig_texts = tokenizer.decode(y_true)

    # Initialize the batch info
    batch_info = defaultdict(int)

    # Print the predictions and process the CER
    for index, (confidence, prediction) in enumerate(y_pred):
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(orig_texts[index])\
            .replace("[UNK]", "�")

        batch_info = process_prediction_type(prediction,
                                             original_text,
                                             batch_info,
                                             do_print=False)

        if config["normalization_file"]:
            normalized_original = loader.normalize(original_text,
                                                   config["normalization_file"]
                                                   )

            # Process the normalized CER
            batch_info = process_prediction_type(prediction,
                                                 normalized_original,
                                                 batch_info,
                                                 do_print=False,
                                                 prefix="Normalized")

        if wbs:
            # Process the WBS CER
            batch_info = process_prediction_type(char_str[index],
                                                 original_text,
                                                 batch_info,
                                                 do_print=False,
                                                 prefix="WBS")

    return batch_info


def perform_test(config: Config,
                 model: tf.keras.Model,
                 test_dataset: DataGenerator,
                 charlist: List[str],
                 dataloader: DataLoader) -> None:
    """
    Performs test run on a dataset using a given model and calculates various
    metrics like Character Error Rate (CER).

    Parameters
    ----------
    config : Config
        A Config object containing arguments for the validation process such as
        mask usage and file paths.
    model : tf.keras.Model
        The Keras model to be validated.
    test_dataset : DataGenerator
        The dataset to be used for testing.
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

    logging.info("Performing test...")

    tokenizer = Tokenizer(charlist, config["use_mask"])
    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(config, charlist, dataloader) \
        if config["corpus_file"] else None

    # Make the predictions
    logging.info("Making predictions...")
    predictions = prediction_model.predict(test_dataset)

    # Initialize the counters
    total_counter = defaultdict(int)

    # Process each batch
    logging.info("Calculating statistics...")
    logging.info("")

    for batch_no, batch in enumerate(test_dataset):
        logging.info(f"Batch {batch_no + 1}/{len(test_dataset)}")

        batch_predictions = predictions[batch_no * config["batch_size"]:
                                        batch_no * config["batch_size"] +
                                        len(batch[0])].numpy()
        batch_true = batch[1]

        batch_info = process_batch(batch_true, batch_predictions, tokenizer,
                                   config, wbs, dataloader, batch_no, charlist)

        # Update the total counter
        for key, value in batch_info.items():
            total_counter[key] += value

    # Define the initial metrics
    metrics = ["CER", "Lower CER", "Simple CER"]

    # Append additional metrics based on conditions
    if config["normalization_file"]:
        metrics += ["Normalized " + m for m in metrics[:3]]
    if wbs:
        metrics += ["WBS " + m for m in metrics[:3]]

    # Calculate CERs
    # Take every third metric (i.e., normalizing, WBS, etc.)
    total_stats = []
    for i in range(0, len(metrics), 3):
        prefix = metrics[i].split(" ")[0] if i > 0 else ""
        total_stats += [*calculate_cers(total_counter, prefix)]

    # Print the final test statistics
    logging.info("")
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final test statistics")
    logging.info("---------------------------")

    # Calculate the CER confidence intervals on all metrics
    intervals = calculate_confidence_intervals(total_stats,
                                               predictions.shape[0])

    # Print the final statistics
    for metric, total_value, interval in zip(metrics, total_stats, intervals):
        logging.info(f"{metric} = {total_value:.4f} +/- {interval:.4f}")

    logging.info(f"Items = {predictions.shape[0]}")
    logging.info("")

    # Output the validation statistics to a csv file
    with open(os.path.join(config["output"], 'test.csv'), 'w') as f:
        header = "cer,cer_lower,cer_simple"
        if config["normalization_file"]:
            header += ",normalized_cer,normalized_cer_lower," \
                "normalized_cer_simple"
        if wbs:
            header += ",wbs_cer,wbs_cer_lower,wbs_cer_simple"

        f.write(header + "\n")
        results = ",".join([str(total_stats[i])
                           for i in range(len(metrics))])
        f.write(results + "\n")
