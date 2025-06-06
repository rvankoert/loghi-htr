# Imports

# > Standard library
import logging
import threading

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from modes.utils import setup_workers, process_batches, output_statistics
from utils.threading import ResultWriter, MetricsCalculator
from utils.wbs import setup_word_beam_search


def perform_evaluation(
    config: Config, model: tf.keras.Model, data_manager: DataManager, mode: str
) -> None:
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
    wbs = (
        setup_word_beam_search(config, tokenizer.token_list)
        if config["corpus_file"]
        else None
    )

    stop_event = threading.Event()

    if mode == "inference":
        # Initialize result writer for inference
        writer = ResultWriter(
            config["results_file"],
            stop_event=stop_event,
            maxsize=config["batch_size"] * 5,
        )
        writer.start()
    else:
        # Initialize metrics calculator for test/validation
        writer = MetricsCalculator(
            config, stop_event=stop_event, log_results=mode == "validation", maxsize=100
        )
        writer.start()

    # Initialize decode workers
    decode_workers = setup_workers(config, data_manager, stop_event, writer.queue, wbs)

    try:
        process_batches(
            dataset, model, config, data_manager, decode_workers, mode, stop_event
        )
    finally:
        # Ensure all workers are stopped gracefully
        for worker in decode_workers:
            worker.stop()
        # wait for all decode_workers to finish
        for worker in decode_workers:
            worker.join()
        writer.stop()

    if mode == "inference":
        # Assert that the number of output lines matches the number of input lines
        input_line_count = len(data_manager.raw_data[mode][0])  # Access the first element of the tuple
        output_line_count = writer.get_written_line_count()
        assert input_line_count == output_line_count, (
            f"Mismatch in line counts: {input_line_count} input lines, "
            f"{output_line_count} output lines"
        )

    if mode in ("test", "validation"):
        output_statistics(writer, config, mode, wbs)