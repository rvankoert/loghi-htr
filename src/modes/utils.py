# > Standard library
import logging
import os
import threading
from typing import List, Optional
from queue import Queue

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.calculate import calc_95_confidence_interval
from utils.threading import DecodingWorker, MetricsCalculator


def setup_workers(
    config: Config,
    data_manager: DataManager,
    stop_event: threading.Event,
    result_queue: Optional[Queue] = None,
    wbs=None,
) -> List[DecodingWorker]:
    """
    Sets up decode workers with optional result writer integration.

    Parameters
    ----------
    config : Config
        The configuration object containing decoding settings, including the number of decoding
        threads.
    data_manager : DataManager
        The data manager providing access to tokenizers and data utilities.
    result_queue : Optional[Queue], optional
        A queue for handling results, by default None.
    wbs : optional
        Optional integration for WBS (if provided), by default None.

    Returns
    -------
    List[DecodingWorker]
        A list of initialized and started decoding workers.
    """
    num_decode_workers: int = config[
        "decoding_threads"
    ]  # Adjust based on available CPU cores
    decode_workers: List[DecodingWorker] = [
        DecodingWorker(
            data_manager.tokenizer,
            config,
            result_queue if result_queue else None,
            stop_event=stop_event,
            wbs=wbs,
            maxsize=5,
        )
        for _ in range(num_decode_workers)
    ]

    for worker in decode_workers:
        worker.start()

    return decode_workers


def process_batches(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    config: Config,
    data_manager: DataManager,
    decode_workers: List[DecodingWorker],
    mode: str,
    stop_event: threading.Event,
):
    """
    Processes batches from the dataset and distributes work to decode workers.

    Parameters
    ----------
    dataset : tf.data.Dataset
        The dataset to be processed, typically yielding batches of input data.
    model : tf.keras.Model
        The model used for generating predictions on the input batches.
    config : Config
        Configuration object containing settings like batch size.
    data_manager : DataManager
        The data manager providing utility functions for accessing filenames and ground truth data.
    decode_workers : List[DecodingWorker]
        List of decode workers to which the processing work will be distributed.
    mode : str
        The mode of processing (e.g., 'inference', 'training').
    stop_event: threading.Event
        The event that signals a worker crash.

    Returns
    -------
    None
    """
    for batch_no, batch in enumerate(dataset):
        if stop_event.is_set():
            logging.error("Stopping inference due to worker thread crash.")
            raise RuntimeError("Worker thread crashed.")

        X = batch[0]
        if mode != "inference":
            y = [
                data_manager.get_ground_truth(mode, i)
                for i in range(
                    batch_no * config["batch_size"],
                    batch_no * config["batch_size"] + len(X),
                )
            ]
        else:
            y = None

        predictions: tf.Tensor = model.predict_on_batch(X)

        batch_filenames: List[str] = [
            data_manager.get_filename(mode, (batch_no * config["batch_size"]) + idx)
            for idx in range(len(predictions))
        ]

        worker_idx: int = batch_no % len(decode_workers)
        decode_workers[worker_idx].input_queue.put(
            (predictions, batch_no, batch_filenames, y)
        )


def output_statistics(
    metrics_calculator: MetricsCalculator, config: Config, mode: str, wbs=None
):
    """
    Logs final statistics and writes them to a CSV file.

    Parameters
    ----------
    metrics_calculator : MetricsCalculator
        Object responsible for computing and storing metrics for evaluation.
    config : Config
        Configuration object containing output directory and optional normalization settings.
    mode : str
        Mode of operation for the statistics (e.g., 'training', 'evaluation').
    wbs : optional
        Optional integration for WBS-specific metrics, by default None.

    Returns
    -------
    None
    """
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final %s statistics", mode)
    logging.info("---------------------------")

    intervals = [
        calc_95_confidence_interval(cer_metric, metrics_calculator.n_items)
        for cer_metric in metrics_calculator.total_stats[:-1]
    ]

    for metric, total_value, interval in zip(
        metrics_calculator.metrics[:-1], metrics_calculator.total_stats[:-1], intervals
    ):
        logging.info("%s = %.4f +/- %.4f", metric, total_value, interval)

    logging.info("Items = %s", metrics_calculator.total_stats[-1])
    logging.info("")

    output_file = os.path.join(config["output"], f"{mode}.csv")
    with open(output_file, "w", encoding="utf-8") as f:
        header = "cer,cer_lower,cer_simple"
        if config["normalization_file"]:
            header += ",normalized_cer,normalized_cer_lower," "normalized_cer_simple"
        if wbs:
            header += ",wbs_cer,wbs_cer_lower,wbs_cer_simple"

        f.write(header + "\n")
        results = ",".join(
            [
                str(metrics_calculator.total_stats[i])
                for i in range(len(metrics_calculator.metrics) - 1)
            ]
        )
        f.write(results + "\n")
