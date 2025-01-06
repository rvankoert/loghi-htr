# Imports

# > Standard library
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from modes.utils import setup_workers, process_batches, output_statistics
from utils.threading import ResultWriter, MetricsCalculator
from utils.wbs import setup_word_beam_search


def perform_evaluation(config: Config,
                       model: tf.keras.Model,
                       data_manager: DataManager,
                       mode: str) -> None:
    """
    Generic evaluation function for inference, test, and validation.

    Parameters
    ----------
    config : Config
        Configuration object.
    model : tf.keras.Model
        Keras model to be evaluated.
    data_manager : DataManager
        Data manager containing datasets and tokenizer.
    mode : str
        Evaluation mode, either 'inference' or 'test'.
    """
    logging.info("Starting %s...", mode)

    dataset = data_manager.datasets[mode]
    tokenizer = data_manager.tokenizer
    wbs = setup_word_beam_search(
        config, tokenizer.token_list) if config["corpus_file"] else None

    if mode == 'inference':
        # Initialize result writer for inference
        writer = ResultWriter(config["results_file"],
                              maxsize=config["batch_size"] * 5)
        writer.start()
    else:
        # Initialize metrics calculator for test/validation
        writer = MetricsCalculator(
            config, log_results=mode == "validation", maxsize=100)
        writer.start()

    # Initialize decode workers
    decode_workers = setup_workers(config, data_manager, writer.queue, wbs)

    try:
        process_batches(dataset, model, config, data_manager,
                        decode_workers, mode)
    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        writer.stop()

    if mode in ('test', 'validation'):
        output_statistics(writer, config, mode, wbs)
