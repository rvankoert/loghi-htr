# Imports
# > Standard library
from collections import defaultdict
import logging
from queue import Queue, Empty
from threading import Thread
from typing import List, Tuple

# > Third-party dependencies
import tensorflow as tf
from word_beam_search import WordBeamSearch

# > Local dependencies
from setup.config import Config
from utils.calculate import (calculate_edit_distances, increment_counters,
                             update_statistics)
from utils.decoding import decode_batch_predictions
from utils.print import print_predictions, display_statistics
from utils.wbs import handle_wbs_results
from utils.text import preprocess_text, normalize_text, Tokenizer


class ResultWriter(Thread):
    """
    Thread dedicated to writing results to a file.

    Parameters
    ----------
    output_file : str
        Path to the file where results will be written.
    maxsize : int, optional
        Maximum size of the queue to limit memory usage, by default 100.

    Attributes
    ----------
    queue : Queue[str]
        Queue for storing result strings.
    output_file : str
        Path to the output file where results will be written.
    running : bool
        Indicates whether the thread is actively processing.
    """

    def __init__(self, output_file: str, maxsize: int = 100):
        super().__init__()
        self.queue: Queue[str] = Queue(maxsize=maxsize)
        self.output_file: str = output_file
        self.running: bool = True

    def run(self) -> None:
        """Main loop of the thread that writes results from the queue to the file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            while self.running or not self.queue.empty():
                try:
                    batch_result: str = self.queue.get(timeout=1.0)
                    for result in batch_result:
                        result_str = \
                            f"{result['filename']}\t{result['confidence']}\t{result['prediction']}"
                        logging.info(result_str)
                        f.write(result_str + "\n")
                    f.flush()
                    self.queue.task_done()
                except Empty:
                    continue

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        self.running = False
        self.join()


class MetricsCalculator(Thread):
    """
    A thread that processes and calculates metrics for model predictions.

    The class runs in a separate thread, processes results from a queue, and calculates
    metrics such as character error rates (CER), word-level error rates, and other statistics
    for a batch of predictions.

    Parameters
    ----------
    config : Config
        Configuration settings for processing, including normalization files.
    maxsize : int, optional
        Maximum size of the input queue to limit memory usage, by default 100.

    Attributes
    ----------
    queue : Queue[str]
        A queue that holds batch results for processing.
    config : Config
        Configuration settings for processing, including normalization files.
    running : bool
        A flag indicating if the thread is still running.
    total_counter : defaultdict[int]
        A counter that accumulates metrics across all batches.
    total_stats : list
        A list that stores cumulative statistics for each batch processed.
    n_items : int
        The total number of items processed across batches.
    metrics : list
        A list of metric names that are calculated and displayed.
    """

    def __init__(self, config: Config, log_results: bool = True, maxsize: int = 100):
        super().__init__()
        self.queue: Queue[str] = Queue(maxsize=maxsize)
        self.config: Config = config
        self.running: bool = True
        self.log_results: bool = log_results

        self.total_counter = defaultdict(int)
        self.total_stats = []
        self.n_items = 0
        self.metrics = []

    def run(self) -> None:
        """Main loop of the thread that processes results from the queue and updates metrics."""
        while self.running or not self.queue.empty():
            try:
                batch_result: str = self.queue.get(timeout=1.0)

                # Initialize the batch counter
                batch_counter = defaultdict(int)

                for result in batch_result:
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    y_true = result["y_true"].replace("[UNK]", "�")
                    filename = result["filename"]
                    char_str = result.get("char_str", None)

                    # Preprocess the text for CER calculation
                    prediction = preprocess_text(prediction)
                    original_text = preprocess_text(y_true)
                    normalized_original = None if not self.config["normalization_file"] else \
                        normalize_text(
                            original_text, self.config["normalization_file"])

                    # Calculate edit distances here so we can use them for printing the
                    # predictions
                    distances = calculate_edit_distances(
                        prediction, original_text)
                    normalized_distances = None if not self.config["normalization_file"] else \
                        calculate_edit_distances(
                            prediction, normalized_original)
                    wbs_distances = None if not char_str else \
                        calculate_edit_distances(char_str, original_text)

                    # Print the predictions if there are any errors
                    if do_print := distances[0] > 0 and self.log_results:
                        print_predictions(filename, original_text, prediction,
                                          normalized_original, char_str)
                        logging.info("Confidence = %.4f", confidence)
                        logging.info("")

                    # Wrap the distances and originals in dictionaries
                    distances_dict = {"distances": distances,
                                      "Normalized distances": normalized_distances,
                                      "WBS distances": wbs_distances}
                    original_dict = {"original": original_text,
                                     "Normalized original": normalized_original,
                                     "WBS original": original_text}

                    # Update the batch counter
                    batch_counter = increment_counters(distances_dict,
                                                       original_dict,
                                                       batch_counter,
                                                       do_print)

                    # Update the total counter
                    self.total_counter = increment_counters(distances_dict,
                                                            original_dict,
                                                            self.total_counter,
                                                            do_print=False)

                # Calculate the CER statistics for this prediction
                self.metrics, batch_stats, self.total_stats = \
                    update_statistics(batch_counter, self.total_counter)

                # Add the number of items to the total tally
                self.n_items += len(batch_result)
                self.metrics.append('Items')
                batch_stats.append(len(batch_result))
                self.total_stats.append(self.n_items)

                # Print batch info
                if self.log_results:
                    display_statistics(
                        batch_stats, self.total_stats, self.metrics)
                    logging.info("")

                self.queue.task_done()
            except Empty:
                continue

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
    wbs : WordBeamSearch
        WordBeamSearch object for decoding predictions.
    """

    def __init__(self, tokenizer: Tokenizer, config: Config, result_queue: Queue,
                 wbs: WordBeamSearch = None, maxsize: int = 10):
        super().__init__()
        self.input_queue: Queue[Tuple[tf.Tensor,
                                      int, List[str]]] = Queue(maxsize=maxsize)
        self.tokenizer = tokenizer
        self.config: Config = config
        self.result_queue: Queue = result_queue
        self.running: bool = True
        self.wbs = wbs

    def run(self) -> None:
        """Main loop of the thread that decodes batches and writes results."""
        while self.running or not self.input_queue.empty():
            try:
                batch_data = self.input_queue.get(timeout=1.0)
                if batch_data is not None:
                    predictions, _, filenames, y_true = batch_data
                    decoded = decode_batch_predictions(
                        predictions,
                        self.tokenizer,
                        self.config["greedy"],
                        self.config["beam_width"]
                    )

                    # Transpose the predictions for WordBeamSearch
                    if self.wbs:
                        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
                        char_str = handle_wbs_results(
                            predsbeam, self.wbs, self.tokenizer.token_list)

                    # Process each prediction and write directly to result writer
                    batch_results = []
                    for idx, (confidence, prediction) in enumerate(decoded):
                        filename = filenames[idx]
                        result = {"filename": filename,
                                  "confidence": confidence,
                                  "prediction": prediction}
                        if y_true is not None:
                            result["y_true"] = y_true[idx]
                        if self.wbs:
                            result["char_str"] = char_str[idx]

                        batch_results.append(result)

                    self.result_queue.put(batch_results)

                self.input_queue.task_done()
            except Empty:
                continue

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        self.running = False
        self.join()
