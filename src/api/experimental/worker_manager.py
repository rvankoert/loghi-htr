# Imports

# > Standard Library
import asyncio
import logging
import multiprocessing as mp
from typing import Any, Dict

# > Local Dependencies
from .workers.decoder_worker import decoder_process_entrypoint
from .workers.predictor_worker import predictor_process_entrypoint

logger = logging.getLogger(__name__)


def start_all_workers(
    app_config: Dict[str, Any],
    stop_event: mp.Event,
    queues: Dict[str, Any],
) -> Dict[str, mp.Process]:
    """
    Start all multiprocessing worker processes (predictor and decoder).

    Parameters
    ----------
    app_config : dict
        Configuration dictionary containing app-wide settings.
    stop_event : mp.Event
        Event used to signal shutdown to worker processes.
    queues : dict
        Dictionary of multiprocessing queues used for inter-process communication.

    Returns
    -------
    dict
        Dictionary mapping worker names to Process instances.
    """
    logger.info("Starting worker processes")

    predictor_args = (
        queues["MPRequest"],
        queues["MPPredictedBatches"],
        queues["MPFinalDecodedResults"],
        app_config,
        stop_event,
    )
    prediction_process = mp.Process(
        target=predictor_process_entrypoint,
        args=predictor_args,
        name="PredictionProcess",
        daemon=True,
    )
    prediction_process.start()
    logger.info(f"Prediction worker started (PID: {prediction_process.pid})")

    decoder_args = (
        queues["MPPredictedBatches"],
        queues["MPFinalDecodedResults"],
        app_config,
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

    return {"Prediction": prediction_process, "Decoding": decoding_process}


def stop_all_workers(workers: Dict[str, mp.Process], stop_event: mp.Event):
    """
    Gracefully stop all running worker processes.

    Parameters
    ----------
    workers : dict
        Dictionary of worker processes.
    stop_event : mp.Event
        Shutdown signal shared with workers.
    """
    logger.info("Signalling worker processes to stop via stop_event")
    if not stop_event.is_set():
        stop_event.set()

    for name, worker in workers.items():
        if worker and worker.is_alive():
            logger.info(
                f"Waiting for worker '{name}' (PID {worker.pid}) to terminate..."
            )
            worker.join(timeout=30)
            if worker.is_alive():
                logger.warning(
                    f"Worker '{name}' (PID {worker.pid}) did not terminate gracefully. Forcing termination."
                )
                worker.terminate()
                worker.join(timeout=5)
            else:
                logger.info(f"Worker '{name}' (PID {worker.pid}) terminated.")
        elif worker:
            logger.info(f"Worker '{name}' (PID {worker.pid}) was already stopped.")
        else:
            logger.warning(f"Worker '{name}' was None, cannot stop.")


async def restart_all_workers(
    app_config: Dict[str, Any],
    stop_event: mp.Event,
    current_workers: Dict[str, mp.Process],
    queues: Dict[str, Any],
    mp_ctx,
) -> Dict[str, mp.Process]:
    """
    Restart all worker processes by stopping current ones and starting new instances.

    Parameters
    ----------
    app_config : dict
        Application configuration.
    stop_event : mp.Event
        Global stop event used to signal all worker shutdown.
    current_workers : dict
        Currently active worker processes.
    queues : dict
        Dictionary of multiprocessing queues used for communication.
    mp_ctx : multiprocessing context
        Multiprocessing context used by the app.

    Returns
    -------
    dict
        New set of worker process instances.
    """
    logger.info("Attempting to restart all workers.")

    # Check if all queues are empty multiple times before restarting workers
    # since the workers might still be processing data while the queues are
    # empty
    empty_count = 0

    while True:
        if all(queue.empty() for queue in queues.values()):
            empty_count += 1

            if empty_count >= 3:
                logger.info("All queues are empty, restarting workers.")

                # Stop all workers
                stop_all_workers(current_workers, stop_event)

                # Clear stop event to allow workers to restart
                stop_event.clear()

                # Restart workers with existing queues
                new_workers = start_all_workers(app_config, stop_event, queues)
                return new_workers
        else:
            empty_count = 0

        # Sleep for a short duration to avoid busy waiting
        await asyncio.sleep(5)
