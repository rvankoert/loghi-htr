# Imports

# > Standard library
import logging
import os
from typing import List

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.calculate import calc_95_confidence_interval
from utils.threading import DecodingWorker, MetricsCalculator
from utils.wbs import setup_word_beam_search


def perform_validation(config: Config,
                       model: tf.keras.Model,
                       data_manager: DataManager) -> None:
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
    data_manager : DataManager
        A DataManager object containing the datasets and tokenizers for
        validation.

    Notes
    -----
    The function processes each batch in the validation dataset, calculates
    CER, and optionally processes Word Beam Search (WBS) results if enabled.
    It also handles the display and logging of statistical information
    throughout the validation process.
    """

    logging.info("Performing validation...")

    tokenizer = data_manager.tokenizer
    validation_dataset = data_manager.datasets['validation']

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(config, tokenizer) \
        if config["corpus_file"] else None

    # Intialize the metric calculator
    metrics_calculator = MetricsCalculator(config, maxsize=10)
    metrics_calculator.start()

    # Initialize decoders with direct access to the result writer
    num_decode_workers: int = 2  # Adjust based on available CPU cores
    decode_workers: List[DecodingWorker] = [
        DecodingWorker(data_manager.tokenizer, config,
                       metrics_calculator.queue, wbs=wbs,
                       maxsize=5)
        for _ in range(num_decode_workers)
    ]

    # Start all decode workers
    for worker in decode_workers:
        worker.start()

    # Process each batch in the validation dataset
    try:
        for batch_no, batch in enumerate(validation_dataset):
            X = batch[0]
            y = [data_manager.get_ground_truth('validation', i)
                 for i in range(batch_no * config["batch_size"],
                                batch_no * config["batch_size"] + len(X))]

            # Get predictions (GPU operation)
            predictions: tf.Tensor = model.predict_on_batch(X)

            # Prepare filenames for the batch
            batch_filenames: List[str] = [
                data_manager.get_filename('validation',
                                          (batch_no * config["batch_size"]) + idx)
                for idx in range(len(predictions))
            ]

            # Distribute work to decode workers
            worker_idx: int = batch_no % num_decode_workers
            decode_workers[worker_idx].input_queue.put(
                (predictions, batch_no, batch_filenames, y)
            )
    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        metrics_calculator.stop()

    # Print the final validation statistics
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final validation statistics")
    logging.info("---------------------------")

    # Calculate the CER confidence intervals on all metrics except Items
    intervals = [calc_95_confidence_interval(cer_metric, metrics_calculator.n_items)
                 for cer_metric in metrics_calculator.total_stats[:-1]]

    # Print the final statistics
    for metric, total_value, interval in zip(metrics_calculator.metrics[:-1],
                                             metrics_calculator.total_stats[:-1],
                                             intervals):
        logging.info("%s = %.4f +/- %.4f", metric, total_value, interval)

    logging.info("Items = %s", metrics_calculator.total_stats[-1])
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
        results = ",".join([str(metrics_calculator.total_stats[i])
                           for i in range(len(metrics_calculator.metrics)-1)])
        f.write(results + "\n")
