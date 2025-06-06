# Imports
# > Standard library
from collections import defaultdict
import logging
from queue import Queue, Empty
import threading
from threading import Thread, Event
from typing import List, Tuple

# > Third-party dependencies
import tensorflow as tf
from word_beam_search import WordBeamSearch

# > Local dependencies
from setup.config import Config
from utils.calculate import (
    calculate_edit_distances,
    increment_counters,
    update_statistics,
)
from utils.decoding import decode_batch_predictions
from utils.print import print_predictions, display_statistics
from utils.wbs import handle_wbs_results
from utils.text import preprocess_text, normalize_text, Tokenizer
import traceback
from bidi.algorithm import get_display


class ResultWriter(Thread):
    def __init__(self, output_file: str, stop_event: Event, maxsize: int = 100):
        super().__init__()
        self.queue: Queue[str] = Queue(maxsize=maxsize)
        self.output_file: str = output_file
        self.running: bool = True
        self.stop_event = stop_event
        self.written_line_count = 0  # Initialize line count

    def run(self) -> None:
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                while self.running and not self.stop_event.is_set():
                    try:
                        batch_result: str = self.queue.get(timeout=1.0)
                        for result in batch_result:
                            result_str = f"{result['filename']}\t{result['confidence']}\t{result['prediction']}"
                            logging.info(result_str)
                            f.write(result_str + "\n")
                            self.written_line_count += 1  # Increment line count
                        f.flush()
                        self.queue.task_done()
                    except Empty:
                        continue
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("ResultWriter encountered an error: %s", str(e))
            self.stop_event.set()
        finally:
            self.stop()

    def get_written_line_count(self) -> int:
        """Returns the number of lines written to the output file."""
        return self.written_line_count

    def stop(self) -> None:
        if self.queue.qsize() > 0:
            logging.warning(
                "Stopping ResultWriter with %d items still in the queue.",
                self.queue.qsize(),
            )
        while not self.queue.empty():
            try:
                batch_result = self.queue.get(timeout=1.0)
                for result in batch_result:
                    result_str = f"{result['filename']}\t{result['confidence']}\t{result['prediction']}"
                    logging.info(result_str)
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(result_str + "\n")
                    self.written_line_count += 1  # Increment line count
                self.queue.task_done()
            except Empty:
                break
        self.running = False
        if self is not threading.current_thread():
            self.join()


class MetricsCalculator(Thread):
    """
    A thread that processes and calculates metrics for model predictions.

    Parameters
    ----------
    config : Config
        Configuration settings for processing, including normalization files.
    log_results : bool, optional
        Whether to log detailed prediction results, by default True.
    maxsize : int, optional
        Maximum size of the input queue, by default 100.

    Attributes
    ----------
    queue : Queue[str]
        Queue holding batch results for processing.
    config : Config
        Configuration settings, including normalization files.
    running : bool
        Flag indicating if the thread is active.
    stop_event: Event
        The event that signals a worker crash.
    total_counter : defaultdict[int]
        Accumulated metrics across all batches.
    total_stats : list
        Cumulative statistics for each processed batch.
    n_items : int
        Total number of processed items.
    metrics : list
        Names of metrics being calculated.
    """

    def __init__(
        self, config, stop_event: Event, log_results: bool = True, maxsize: int = 100
    ):
        super().__init__()
        self.queue = Queue(maxsize=maxsize)
        self.config = config
        self.running = True
        self.stop_event = stop_event
        self.log_results = log_results

        self.total_counter = defaultdict(int)
        self.total_stats = []
        self.n_items = 0
        self.metrics = []

    def run(self) -> None:
        """Main loop that processes results from the queue and updates metrics."""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    batch_result = self.queue.get(timeout=1.0)
                    self._process_batch(batch_result)
                    self.queue.task_done()
                except Empty:
                    continue
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("MetricsCalculator encountered an error: %s", str(e))
            self.stop_event.set()
        finally:
            self.running = False

    def _process_batch(self, batch_result: list) -> None:
        """Processes a batch of predictions and updates counters and metrics."""
        batch_counter = defaultdict(int)

        for result in batch_result:
            prediction = preprocess_text(result["prediction"])
            original_text = preprocess_text(result["y_true"])
            if self.config["bidirectional"]:
                prediction = get_display(prediction)
                original_text = get_display(original_text)

            normalized_original = (
                normalize_text(original_text, self.config["normalization_file"])
                if self.config["normalization_file"]
                else None
            )
            char_str = result.get("char_str", None)
            distances = calculate_edit_distances(prediction, original_text)
            normalized_distances = (
                calculate_edit_distances(prediction, normalized_original)
                if normalized_original
                else None
            )
            wbs_distances = (
                calculate_edit_distances(char_str, original_text) if char_str else None
            )

            if do_print := distances[0] > 0 and self.log_results:
                print_predictions(
                    result["filename"],
                    original_text,
                    prediction,
                    normalized_original,
                    char_str,
                )
                logging.info("Confidence = %.4f", result["confidence"])
                logging.info("")

            distances_dict = {
                "distances": distances,
                "Normalized distances": normalized_distances,
                "WBS distances": wbs_distances,
            }
            original_dict = {
                "original": original_text,
                "Normalized original": normalized_original,
                "WBS original": original_text,
            }

            batch_counter = increment_counters(
                distances_dict, original_dict, batch_counter, do_print=do_print
            )
            self.total_counter = increment_counters(
                distances_dict, original_dict, self.total_counter, do_print=False
            )

        self._update_statistics(batch_counter, len(batch_result))

    def _update_statistics(self, batch_counter: defaultdict, batch_size: int) -> None:
        """Updates statistics and optionally logs them."""
        self.metrics, batch_stats, self.total_stats = update_statistics(
            batch_counter, self.total_counter
        )
        self.n_items += batch_size
        self.metrics.append("Items")
        batch_stats.append(batch_size)
        self.total_stats.append(self.n_items)

        if self.log_results:
            display_statistics(batch_stats, self.total_stats, self.metrics)
            logging.info("")

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        self.running = False
        self.join()


class DecodingWorker(Thread):
    """
    Thread dedicated to CTC decoding.

    Parameters
    ----------
    tokenizer : Tokenizer
        Tokenizer object for decoding predictions.
    config : Config
        Configuration object containing decoding parameters.
    result_queue : Queue[List[Dict[str, Union[str, float]]]]
        Queue for storing decoded results.
    wbs : WordBeamSearch, optional
        WordBeamSearch object for decoding predictions, by default None.
    maxsize : int, optional
        Maximum size of the input queue to limit memory usage, by default 10.

    Attributes
    ----------
    input_queue : Queue[Tuple[tf.Tensor, int, List[str]]]
        Queue for receiving batches of data.
    tokenizer : object
        Tokenizer object for decoding predictions.
    config : Config
        Configuration object for decoding parameters.
    result_queue : Queue[List[Dict[str, Union[str, float]]]]
        Queue for storing decoded results.
    running : bool
        Indicates whether the thread is actively processing.
    stop_event: Event
        The event that signals a worker crash.
    wbs : WordBeamSearch
        WordBeamSearch object for decoding predictions.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        config: Config,
        result_queue: Queue,
        stop_event: Event,
        wbs: WordBeamSearch = None,
        maxsize: int = 10,
    ):
        super().__init__()
        self.input_queue: Queue[Tuple[tf.Tensor, int, List[str]]] = Queue(
            maxsize=maxsize
        )
        self.tokenizer = tokenizer
        self.config: Config = config
        self.result_queue: Queue = result_queue
        self.running: bool = True
        self.stop_event = stop_event
        self.wbs = wbs

    def run(self) -> None:
        """Main loop of the thread that decodes batches and writes results."""
        try:
            while self.running and not self.stop_event.is_set():
                try:
                    batch_data = self.input_queue.get(timeout=1.0)
                    if batch_data is not None:
                        predictions, _, filenames, y_true = batch_data
                        decoded = decode_batch_predictions(
                            predictions,
                            self.tokenizer,
                            self.config["greedy"],
                            self.config["beam_width"],
                        )

                        # Transpose the predictions for WordBeamSearch
                        if self.wbs:
                            predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
                            char_str = handle_wbs_results(
                                predsbeam, self.wbs, self.tokenizer.token_list
                            )

                        # Process each prediction and write directly to result writer
                        batch_results = []
                        for idx, (confidence, prediction) in enumerate(decoded):
                            filename = filenames[idx]
                            result = {
                                "filename": filename,
                                "confidence": confidence,
                                "prediction": prediction,
                            }
                            if y_true is not None:
                                result["y_true"] = y_true[idx]
                            if self.wbs:
                                result["char_str"] = char_str[idx]

                            batch_results.append(result)

                        self.result_queue.put(batch_results)

                    self.input_queue.task_done()
                except Empty:
                    continue
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error("DecodingWorker encountered an error: %s", str(e))
            self.stop_event.set()
        finally:
            self.stop()

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        if self.input_queue.qsize() > 0:
            logging.warning(
                "Stopping DecodingWorker with %d items still in the queue.",
                self.input_queue.qsize(),
            )
        while not self.input_queue.empty():
            try:
                batch_data = self.input_queue.get(timeout=1.0)
                if batch_data is not None:
                    predictions, _, filenames, y_true = batch_data
                    decoded = decode_batch_predictions(
                        predictions,
                        self.tokenizer,
                        self.config["greedy"],
                        self.config["beam_width"],
                    )

                    # Transpose the predictions for WordBeamSearch
                    if self.wbs:
                        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
                        char_str = handle_wbs_results(
                            predsbeam, self.wbs, self.tokenizer.token_list
                        )

                    # Process each prediction and write directly to result writer
                    batch_results = []
                    for idx, (confidence, prediction) in enumerate(decoded):
                        filename = filenames[idx]
                        result = {
                            "filename": filename,
                            "confidence": confidence,
                            "prediction": prediction,
                        }
                        if y_true is not None:
                            result["y_true"] = y_true[idx]
                        if self.wbs:
                            result["char_str"] = char_str[idx]

                        batch_results.append(result)

                    self.result_queue.put(batch_results)

                self.input_queue.task_done()
            except Empty:
                break
        if self.input_queue.qsize() > 0:
            logging.warning(
                "DecodingWorker stopped with %d items still in the queue.",
                self.input_queue.qsize(),
            )
        # Clean up the worker
        self.running = False
        if self is not threading.current_thread():
            self.join()
