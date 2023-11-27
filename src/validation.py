# Imports

# > Standard library
from collections import defaultdict
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from model_management import get_prediction_model
from utils import decode_batch_predictions, Utils, normalize_confidence
from utility_functions import calculate_confidence_intervals, \
    calculate_edit_distances, display_statistics, preprocess_text, \
    process_cer_type, process_prediction_type, print_predictions
from wbs_utils import setup_word_beam_search, handle_wbs_results


def process_batch(batch, prediction_model, utils_object,
                  args, wbs, loader, batch_no, chars):
    X, y_true = batch

    # Get the predictions
    predictions = prediction_model.predict(X, verbose=0)
    y_pred = decode_batch_predictions(
        predictions, utils_object, args.greedy,
        args.beam_width, args.num_oov_indices)[0]

    # Transpose the predictions for WordBeamSearch
    if wbs:
        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
        char_str = handle_wbs_results(predsbeam, wbs, args, chars)
    else:
        char_str = None

    # Get the original texts
    orig_texts = [tf.strings.reduce_join(utils_object.num_to_char(label))
                  .numpy().decode("utf-8").strip() for label in y_true]

    # Initialize the batch info
    batch_info = defaultdict(int)

    # Print the predictions and process the CER
    for index, (confidence, prediction) in enumerate(y_pred):
        # Normalize the confidence before processing because it was determined
        # on the original prediction
        normalized_confidence = normalize_confidence(confidence, prediction)

        # Preprocess the text for CER calculation
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(orig_texts[index])

        # Calculate edit distances here so we can use them for printing the
        # predictions
        distances = \
            calculate_edit_distances(prediction, original_text)
        do_print = distances[0] > 0

        # Print the predictions if there are any errors
        if do_print:
            filename = loader.get_item('validation',
                                       (batch_no * args.batch_size) + index)
            wbs_str = char_str[index] if wbs else None
            print_predictions(filename, original_text,
                              prediction, wbs_str)
            logging.info(f"Confidence = {normalized_confidence:.4f}")
            logging.info("")

        batch_info = process_prediction_type(prediction,
                                             original_text,
                                             batch_info,
                                             do_print)

        if args.normalization_file:
            # Normalize the text
            normalized_prediction = loader.normalize(prediction,
                                                     args.normalization_file)
            normalized_original = loader.normalize(original_text,
                                                   args.normalization_file)

            # Process the normalized CER
            batch_info = process_prediction_type(normalized_prediction,
                                                 normalized_original,
                                                 batch_info,
                                                 do_print,
                                                 prefix="Normalized")

        if wbs:
            # Process the WBS CER
            batch_info = process_prediction_type(char_str[index],
                                                 original_text,
                                                 batch_info,
                                                 do_print,
                                                 prefix="WBS")

    return batch_info


def perform_validation(args, model, validation_dataset, char_list, dataloader):
    logging.info("Performing validation...")

    utils_object = Utils(char_list, args.use_mask)
    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(args, char_list, dataloader) \
        if args.corpus_file else None

    # Initialize variables for CER calculation
    n_items = 0
    total_counter = defaultdict(int)
    total_normalized_counter = defaultdict(int)
    total_wbs_counter = defaultdict(int)

    # Process each batch in the validation dataset
    for batch_no, batch in enumerate(validation_dataset):
        # Logic for processing each batch, calculating CER, etc.
        batch_info = process_batch(batch, prediction_model, utils_object, args,
                                   wbs, dataloader, batch_no, char_list)
        metrics, batch_stats, total_stats = [], [], []

        # Calculate the CER
        total_counter, metrics, batch_stats, total_stats \
            = process_cer_type(
                batch_info, total_counter, metrics, batch_stats, total_stats)

        # Calculate the normalized CER
        if args.normalization_file:
            total_normalized_counter, metrics, batch_stats, total_stats \
                = process_cer_type(
                    batch_info, total_normalized_counter, metrics, batch_stats,
                    total_stats, prefix="Normalized")

        # Calculate the WBS CER
        if wbs:
            total_wbs_counter, metrics, batch_stats, total_stats \
                = process_cer_type(
                    batch_info, total_wbs_counter, metrics, batch_stats,
                    total_stats, prefix="WBS")

        # Print batch info
        n_items += len(batch[1])
        metrics.append('Items')
        batch_stats.append(len(batch[1]))
        total_stats.append(n_items)

        display_statistics(batch_stats, total_stats, metrics)
        logging.info("")

    # Print the final validation statistics
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final validation statistics")
    logging.info("---------------------------")

    # Calculate the CER confidence intervals on all metrics except Items
    intervals = calculate_confidence_intervals(total_stats[:-1], n_items)

    # Print the final statistics
    for metric, total_value, interval in zip(metrics[:-1], total_stats[:-1],
                                             intervals):
        logging.info(f"{metric} = {total_value:.4f} +/- {interval:.4f}")

    logging.info(f"Items = {total_stats[-1]}")
    logging.info("")
