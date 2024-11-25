# Imports
# > Standard library
from typing import List

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.threading import DecodingWorker, ResultWriter


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
                       result_writer.queue, maxsize=5)
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
                (predictions, batch_no, batch_filenames, None)
            )

    finally:
        # Clean up workers
        for worker in decode_workers:
            worker.stop()
        result_writer.stop()
