import multiprocessing as mp
import logging
from typing import Dict, Any, List

# Relative imports to sibling worker modules
from .workers.predictor_worker import predictor_process_entrypoint
from .workers.decoder_worker import decoder_process_entrypoint

logger = logging.getLogger(__name__)


def start_all_workers(
    app_config: Dict[str, Any],  # Main application config
    stop_event: mp.Event,
    queues: Dict[str, Any],
) -> Dict[str, mp.Process]:
    """Initializes and starts all worker processes."""

    logger.info("Starting worker processes")

    # Predictor worker
    predictor_args = (
        queues["MPRequest"],
        queues["MPPredictedBatches"],
        app_config,  # Pass relevant parts of app_config
        stop_event,
    )
    prediction_process = mp.Process(
        target=predictor_process_entrypoint,
        args=predictor_args,
        name="PredictionProcess",
        daemon=True,  # Ensure daemonic processes if they should not block app exit
    )
    prediction_process.start()
    logger.info(f"Prediction worker started (PID: {prediction_process.pid})")

    # Decoder worker
    decoder_args = (
        queues["MPPredictedBatches"],
        queues["MPFinalDecodedResults"],
        app_config,  # Pass relevant parts of app_config
        stop_event,
    )
    decoding_process = mp.Process(
        target=decoder_process_entrypoint,
        args=decoder_args,
        name="DecodingProcess",
        daemon=True,
    )
    decoding_process.start()
    logger.info(f"Decoding worker started (PID: {decoding_process.pid})")

    workers = {"Prediction": prediction_process, "Decoding": decoding_process}
    return workers


def stop_all_workers(workers: Dict[str, mp.Process], stop_event: mp.Event):
    """Stops all worker processes gracefully."""
    logger.info("Signalling worker processes to stop via stop_event")
    if not stop_event.is_set():  # Set event if not already set
        stop_event.set()

    # Additionally, send sentinel (None) to input queues if workers might be blocked on get()
    # This assumes workers' input queues are known and they handle None gracefully.
    # For predictor: queues["MPRequest"]
    # For decoder: queues["MPPredictedBatches"]
    # This step can be crucial if workers don't check stop_event frequently enough when blocked on queue.get().
    # Example:
    # if "MPRequest" in queues: queues["MPRequest"].put(None) # May need to do this carefully if queue is full
    # if "MPPredictedBatches" in queues: queues["MPPredictedBatches"].put(None)

    for name, worker in workers.items():
        if worker and worker.is_alive():  # Check if worker is not None and is alive
            logger.info(
                f"Waiting for worker '{name}' (PID {worker.pid}) to terminate..."
            )
            worker.join(timeout=30)  # Increased timeout
            if worker.is_alive():
                logger.warning(
                    f"Worker '{name}' (PID {worker.pid}) did not terminate gracefully. Forcing termination."
                )
                worker.terminate()  # Force terminate if join times out
                worker.join(timeout=5)  # Wait for termination
            else:
                logger.info(f"Worker '{name}' (PID {worker.pid}) terminated.")
        elif worker:
            logger.info(f"Worker '{name}' (PID {worker.pid}) was already stopped.")
        else:
            logger.warning(f"Worker '{name}' was None, cannot stop.")


async def restart_all_workers(
    app_config: Dict[str, Any],
    stop_event: mp.Event,  # The global stop event
    current_workers: Dict[str, mp.Process],
    queues: Dict[str, Any],
    mp_ctx,  # Multiprocessing context
) -> Dict[str, mp.Process]:
    """Restarts all worker processes."""
    logger.info("Attempting to restart all workers.")

    # 1. Signal existing workers and bridges to stop using the existing stop_event
    if not stop_event.is_set():
        logger.info("Setting stop event to halt current workers and bridges.")
        stop_event.set()

    # (Bridges are stopped separately in app.py's monitor_memory before calling this)

    # 2. Stop existing workers
    stop_all_workers(current_workers, stop_event)  # This will use the already set event

    # 3. Create a new stop_event for the new set of workers & bridges
    #    Or, clear and reuse the existing one if app logic allows.
    #    The original code suggests stop_event is cleared and reused.
    logger.info("Clearing stop event for new workers and bridges.")
    stop_event.clear()

    # MP Queues: The problem description implies queues are somewhat persistent or
    # that the app reuses the `queues` dict.
    # If MP queues need to be drained or re-created, that logic would go here.
    # The original `restart_workers` waited for MP queues to be empty.
    # This version assumes that stopping workers (which should drain their input queues
    # if they process remaining items after stop_event or receive a sentinel)
    # and then restarting them with the *same* queue instances is the desired behavior.
    # If clean MP queues are needed, they'd have to be re-initialized and app.state.queues updated.

    # Let's add the queue check from original code for robustness:
    mp_queues_to_check = [
        queues["MPRequest"],
        queues["MPPredictedBatches"],
        # queues["MPFinalDecodedResults"], # This is output of decoder, less critical to be empty *before* restart
    ]

    logger.info("Checking if critical MP queues are empty before restarting workers...")
    for q_name, q_instance in {
        "MPRequest": queues["MPRequest"],
        "MPPredictedBatches": queues["MPPredictedBatches"],
    }.items():
        # Try to clear by getting items non-blockingly
        cleared_count = 0
        try:
            while not q_instance.empty():
                q_instance.get_nowait()
                cleared_count += 1
        except Exception:  # Catches Empty exception too
            pass
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} items from {q_name} during restart.")
        if not q_instance.empty():
            logger.warning(
                f"Queue {q_name} is not empty after attempting to clear. Size: ~{q_instance.qsize()}. Proceeding with restart."
            )
        else:
            logger.info(f"Queue {q_name} is empty.")

    # 4. Start new worker processes
    logger.info("Starting new set of worker processes.")
    new_workers = start_all_workers(app_config, stop_event, queues)

    logger.info("Workers restarted successfully.")
    return new_workers
