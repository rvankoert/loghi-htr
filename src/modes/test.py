# > Standard library
from collections import defaultdict
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.generator import DataGenerator
from data.loader import DataLoader
from model.management import get_prediction_model
from setup.config import Config
from utils.calculate import calculate_confidence_intervals, \
    process_cer_type, process_prediction_type
from utils.decoding import decode_batch_predictions
from utils.text import preprocess_text, Tokenizer
from utils.wbs import setup_word_beam_search, handle_wbs_results


def process_batch(batch: Tuple[tf.Tensor, tf.Tensor],
                  prediction_model: tf.keras.Model,
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
    batch : Tuple[tf.Tensor, tf.Tensor]
        A tuple containing the input data (X) and true labels (y_true) for the
        batch.
    prediction_model : tf.keras.Model
        The prediction model derived from the main model for inference.
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

    args = config.args

    X, y_true = batch

    # Get the predictions
    predictions = prediction_model.predict_on_batch(X)
    y_pred = decode_batch_predictions(predictions, tokenizer, args.greedy,
                                      args.beam_width)

    # Transpose the predictions for WordBeamSearch
    if wbs:
        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
        char_str = handle_wbs_results(predsbeam, wbs, args, chars)
    else:
        char_str = None

    # Get the original texts
    orig_texts = tokenizer.decode(y_true)

    # Initialize the batch info
    batch_info = defaultdict(int)

    # Print the predictions and process the CER
    for index, (confidence, prediction) in enumerate(y_pred):
        # Preprocess the text for CER calculation
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(orig_texts[index])\
            .replace("[UNK]", "�")
        normalized_original = None if not args.normalization_file else \
            loader.normalize(original_text, args.normalization_file)

        batch_info = process_prediction_type(prediction,
                                             original_text,
                                             batch_info,
                                             do_print=False)

        if args.normalization_file:
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

    args = config.args

    tokenizer = Tokenizer(charlist, args.use_mask)
    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(config, charlist, dataloader) \
        if args.corpus_file else None

    # Initialize variables for CER calculation
    n_items = 0
    total_counter = defaultdict(int)
    total_normalized_counter = defaultdict(int)
    total_wbs_counter = defaultdict(int)

    # Process each batch in the test dataset
    for batch_no, batch in enumerate(test_dataset):
        logging.info(f"Batch {batch_no+1}/{len(test_dataset)}")

        # Logic for processing each batch, calculating CER, etc.
        batch_info = process_batch(batch, prediction_model, tokenizer, config,
                                   wbs, dataloader, batch_no, charlist)
        metrics, batch_stats, total_stats = [], [], []

        # Calculate the CER
        total_counter, metrics, batch_stats, total_stats\
            = process_cer_type(batch_info, total_counter, metrics,
                               batch_stats, total_stats)

        # Calculate the normalized CER
        if args.normalization_file:
            total_normalized_counter, metrics, batch_stats, total_stats\
                = process_cer_type(batch_info, total_normalized_counter,
                                   metrics, batch_stats, total_stats,
                                   prefix="Normalized")

        # Calculate the WBS CER
        if wbs:
            total_wbs_counter, metrics, batch_stats, total_stats\
                = process_cer_type(batch_info, total_wbs_counter, metrics,
                                   batch_stats, total_stats, prefix="WBS")

        # Print batch info
        n_items += len(batch[1])
        metrics.append('Items')
        batch_stats.append(len(batch[1]))
        total_stats.append(n_items)

    # Print the final test statistics
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final test statistics")
    logging.info("---------------------------")

    # Calculate the CER confidence intervals on all metrics except Items
    intervals = calculate_confidence_intervals(total_stats[:-1], n_items)

    # Print the final statistics
    for metric, total_value, interval in zip(metrics[:-1], total_stats[:-1],
                                             intervals):
        logging.info(f"{metric} = {total_value:.4f} +/- {interval:.4f}")

    logging.info(f"Items = {total_stats[-1]}")
    logging.info("")

    # Output the validation statistics to a csv file
    with open(os.path.join(args.output, 'test.csv'), 'w') as f:
        header = "cer,cer_lower,cer_simple"
        if args.normalization_file:
            header += ",normalized_cer,normalized_cer_lower,"
            "normalized_cer_simple"
        if wbs:
            header += ",wbs_cer,wbs_cer_lower,wbs_cer_simple"

        f.write(header + "\n")
        results = ",".join([str(total_stats[i])
                           for i in range(len(metrics)-1)])
        f.write(results + "\n")
