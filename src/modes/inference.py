# Imports
# > Standard library
import logging
from queue import Queue, Empty
from threading import Thread
from typing import List, Tuple, Optional

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.decoding import decode_batch_predictions


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
                    result: str = self.queue.get(timeout=1.0)
                    f.write(result + "\n")
                    f.flush()
                    self.queue.task_done()
                except Empty:
                    continue

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        self.running = False
        self.join()


class DecodingWorker(Thread):
    """
    Thread dedicated to CTC decoding and directly writing results.

    Parameters
    ----------
    tokenizer : object
        Tokenizer object for decoding predictions.
    config : Config
        Configuration object containing decoding parameters.
    result_writer : ResultWriter
        Thread responsible for writing results to a file.
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
    result_writer : ResultWriter
        Thread responsible for writing results to a file.
    running : bool
        Indicates whether the thread is actively processing.
    """

    def __init__(self, tokenizer, config: Config, result_writer: ResultWriter, maxsize: int = 10):
        super().__init__()
        self.input_queue: Queue[Tuple[tf.Tensor,
                                      int, List[str]]] = Queue(maxsize=maxsize)
        self.tokenizer = tokenizer
        self.config: Config = config
        self.result_writer: ResultWriter = result_writer
        self.running: bool = True

    def run(self) -> None:
        """Main loop of the thread that decodes batches and writes results."""
        while self.running or not self.input_queue.empty():
            try:
                batch_data: Optional[Tuple[tf.Tensor, int, List[str]]] = self.input_queue.get(
                    timeout=1.0)
                if batch_data is not None:
                    predictions, _, filenames = batch_data
                    decoded = decode_batch_predictions(
                        predictions,
                        self.tokenizer,
                        self.config["greedy"],
                        self.config["beam_width"]
                    )

                    # Process each prediction and write directly to result writer
                    for idx, (confidence, prediction) in enumerate(decoded):
                        filename = filenames[idx]
                        result_str = f"{filename}\t{confidence}\t{prediction}"
                        logging.info(result_str)
                        self.result_writer.queue.put(result_str)

                self.input_queue.task_done()
            except Empty:
                continue

    def stop(self) -> None:
        """Stops the thread and waits for it to terminate."""
        self.running = False
        self.join()


def perform_inference(config: Config,
                      model: tf.keras.Model,
                      inference_dataset: tf.data.Dataset,
                      data_manager: DataManager) -> None:
    """
    Performs inference using parallel processing with direct communication between
    decoders and the result writer.

    Parameters
    ----------
    config : Config
        Configuration object containing parameters for inference.
    model : tf.keras.Model
        Trained TensorFlow model for making predictions.
    inference_dataset : tf.data.Dataset
        Dataset for performing inference.
    data_manager : DataManager
        Object managing dataset and filename retrieval.

    Notes
    -----
    This optimized implementation:
        - Uses separate threads for CTC decoding and result writing.
        - Implements direct communication between decoders and the result writer.
        - Prefetches the next batch while processing the current one.
        - Minimizes GPU blocking time.
    """
    # Initialize result writer first
    result_writer = ResultWriter(config["results_file"],
                                 maxsize=config["batch_size"] * 5)
    result_writer.start()

    # Initialize decoders with direct access to the result writer
    num_decode_workers: int = 2  # Adjust based on available CPU cores
    decode_workers: List[DecodingWorker] = [
        DecodingWorker(data_manager.tokenizer, config,
                       result_writer, maxsize=5)
        for _ in range(num_decode_workers)
    ]

    # Start all decode workers
    for worker in decode_workers:
        worker.start()

    try:
        for batch_no, batch in enumerate(inference_dataset):
            # Get predictions (GPU operation)
            predictions: tf.Tensor = model.predict_on_batch(batch[0])

            # Prepare filenames for the batch
            batch_filenames: List[str] = [
                data_manager.get_filename('inference',
                                          (batch_no * config["batch_size"]) + idx)
                for idx in range(len(predictions))
            ]

            # Distribute work to decode workers
            worker_idx: int = batch_no % num_decode_workers
            decode_workers[worker_idx].input_queue.put(
                (predictions, batch_no, batch_filenames)
            )

    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        result_writer.stop()
