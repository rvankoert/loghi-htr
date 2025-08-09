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
    num_decode_workers: int = 1
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


def process_batches(dataset, model, config, data_manager, decode_workers, mode, stop_event):
    """
    Process batches from the dataset, perform predictions, and send results to workers.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset to process.
    model : tf.keras.Model
        Keras model to perform predictions.
    config : dict
        Configuration dictionary.
    data_manager : DataManager
        Data manager containing datasets and tokenizer.
    decode_workers : List[DecodingWorker]
        List of workers for decoding predictions.
    mode : str
        Evaluation mode, either 'inference' or 'test'/'validation'.
    stop_event : threading.Event
        Event to signal processing should stop.
    """
    logging.info("Processing batches for %s...", mode)

    try:
        # Determine if we need ground truth (not in inference mode)
        need_ground_truth = mode != "inference"
        worker_idx = 0
        num_workers = len(decode_workers)

        # Process each batch
        for i, batch in enumerate(dataset):
            if stop_event.is_set():
                logging.warning("Processing stopped due to worker error")
                break

            # Unpack the batch
            images, labels, sample_weights = batch
            current_batch_size = tf.shape(images)[0]

            # Get filenames and ground truth if needed
            filenames = [data_manager.get_filename(mode, i * config["batch_size"] + j)
                         for j in range(current_batch_size)]

            y_true = None
            if need_ground_truth:
                y_true = [data_manager.get_ground_truth(mode, i * config["batch_size"] + j)
                          for j in range(current_batch_size)]

            # Perform prediction with error handling
            try:
                predictions = model.predict_on_batch(images)

                # Ensure predictions and filenames have matching length
                if len(predictions) > len(filenames):
                    logging.warning(f"Truncating predictions to match filenames: {len(predictions)} → {len(filenames)}")
                    predictions = predictions[:len(filenames)]

                # Distribute work to decode workers in a round-robin fashion
                decode_workers[worker_idx].input_queue.put((predictions, current_batch_size, filenames, y_true))
                worker_idx = (worker_idx + 1) % num_workers

                # Log progress periodically
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1} batches...")

            except tf.errors.OutOfRangeError as e:
                logging.error(f"Out of range error on batch {i}: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Error processing batch {i}: {str(e)}")
                if not config.get("continue_on_error", False):
                    stop_event.set()
                    break
                continue
    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}")
        stop_event.set()

    logging.info("Finished processing batches.")


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
