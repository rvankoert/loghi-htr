# Imports

# > Standard library
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from modes.utils import setup_workers, process_batches
from utils.threading import ResultWriter
from utils.wbs import setup_word_beam_search


def perform_inference(config: Config,
                      model: tf.keras.Model,
                      data_manager: DataManager) -> None:
    """
    Generic evaluation function for inference.

    Parameters
    ----------
    config : Config
        Configuration object.
    model : tf.keras.Model
        Keras model to be inferenced on.
    data_manager : DataManager
        Data manager containing datasets and tokenizer.
    """
    logging.info("Starting %s...", 'inference')

    dataset = data_manager.datasets['inference']
    tokenizer = data_manager.tokenizer
    wbs = setup_word_beam_search(
        config, tokenizer.token_list) if config["corpus_file"] else None

    # Initialize result writer
    result_writer = ResultWriter(config["results_file"],
                                 maxsize=config["batch_size"] * 5)
    result_writer.start()

    # Initialize decode workers
    decode_workers = setup_workers(config, data_manager,
                                   result_writer.queue, wbs)

    try:
        process_batches(dataset, model, config, data_manager,
                        decode_workers, 'inference')
    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        result_writer.stop()
