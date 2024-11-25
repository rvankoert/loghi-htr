# Imports

# > Standard library
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from modes.utils import setup_workers, process_batches, output_statistics
from utils.threading import MetricsCalculator
from utils.wbs import setup_word_beam_search


def perform_test(config: Config,
                 model: tf.keras.Model,
                 data_manager: DataManager):
    """
    Generic evaluation function for test and validation.

    Parameters
    ----------
    config : Config
        Configuration object.
    model : tf.keras.Model
        Keras model to be evaluated.
    data_manager : DataManager
        Data manager containing datasets and tokenizers.
    """
    logging.info("Performing %s...", 'test')

    dataset = data_manager.datasets['test']
    tokenizer = data_manager.tokenizer
    wbs = setup_word_beam_search(
        config, tokenizer.token_list) if config["corpus_file"] else None

    metrics_calculator = MetricsCalculator(config, log_results=False,
                                           maxsize=100)
    metrics_calculator.start()
    decode_workers = setup_workers(config, data_manager,
                                   metrics_calculator.queue, wbs)

    try:
        process_batches(dataset, model, config, data_manager,
                        decode_workers, 'test')
    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        metrics_calculator.stop()

    output_statistics(metrics_calculator, config, 'test', wbs)
