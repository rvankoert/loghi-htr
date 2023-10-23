# Imports

# > Standard library
import logging
from multiprocessing import Manager, Process

# > Local dependencies
import errors
from routes import main
from app_utils import setup_logging, get_env_variable
from batch_predictor import batch_prediction_worker
from image_preparator import image_preparation_worker

# > Third-party dependencies
from flask import Flask


def create_app(model_path: str,
               charlist_path: str,
               batch_size: int,
               output_path: str,
               gpus: str,
               num_channels: int,
               max_queue_size: int) -> Flask:
    """
    Create and configure a Flask app for image prediction.

    This function initializes a Flask app, sets up necessary configurations,
    starts image preparation and batch prediction processes, and returns the
    configured app instance.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file.
    charlist_path : str
        Path to the character list file.
    batch_size : int
        Number of images to process in a batch.
    output_path : str
        Path where predictions should be saved.
    gpus : str
        IDs of GPUs to be used (comma-separated).
    num_channels : int
        Number of channels desired for the input images (e.g., 1 for grayscale,
        3 for RGB).
    max_queue_size : int
        Maximum number of images to be stored in the request queue. Increasing
        this number will increase the memory usage.

    Returns
    -------
    Flask
        Configured Flask app instance ready for serving.

    Side Effects
    ------------
    - Initializes and starts image preparation and batch prediction processes.
    - Logs various messages regarding the app and process initialization.
    """

    logger = logging.getLogger(__name__)

    # Create Flask app
    logger.info("Creating Flask app")
    app = Flask(__name__)

    # Register error handler
    app.register_error_handler(ValueError, errors.handle_invalid_usage)
    app.register_error_handler(405, errors.method_not_allowed)

    # Register blueprints
    app.register_blueprint(main)

    # Create a thread-safe Queue
    logger.info("Initializing request queue")
    manager = Manager()
    app.request_queue = manager.JoinableQueue(maxsize=max_queue_size//2)

    # Max size of prepared queue is half of the max size of request queue
    # expressed in number of batches
    max_prepared_queue_size = max_queue_size // 2 // batch_size
    app.prepared_queue = manager.JoinableQueue(maxsize=max_prepared_queue_size)

    # Start the image preparation process
    logger.info("Starting image preparation process")
    app.preparation_process = Process(
        target=image_preparation_worker,
        args=(batch_size, app.request_queue,
              app.prepared_queue, num_channels),
        name="Image Preparation Process")
    app.preparation_process.start()

    # Start the batch prediction process
    logger.info("Starting batch prediction process")
    app.prediction_process = Process(
        target=batch_prediction_worker,
        args=(app.prepared_queue, model_path,
              charlist_path, output_path, num_channels, gpus),
        name="Batch Prediction Process")
    app.prediction_process.start()

    return app


if __name__ == '__main__':
    # Set up logging
    logger = setup_logging("INFO")

    # Get Loghi-HTR options from environment variables
    logger.info("Getting Loghi-HTR options from environment variables")
    model_path = get_env_variable("LOGHI_MODEL_PATH")
    charlist_path = get_env_variable("LOGHI_CHARLIST_PATH")
    batch_size = int(get_env_variable("LOGHI_BATCH_SIZE", "256"))
    output_path = get_env_variable("LOGHI_OUTPUT_PATH")
    max_queue_size = int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000"))
    num_channels = int(get_env_variable("LOGHI_MODEL_CHANNELS"))

    # Get GPU options from environment variables
    logger.info("Getting GPU options from environment variables")
    gpus = get_env_variable("LOGHI_GPUS", "0")

    app = create_app(
        model_path,
        charlist_path,
        batch_size,
        output_path,
        gpus,
        num_channels,
        max_queue_size)
    app.run(debug=True)
