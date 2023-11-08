# Imports

# > Standard library
import logging

# > Local dependencies
import errors
from routes import main
from app_utils import setup_logging, get_env_variable, start_processes

# > Third-party dependencies
from flask import Flask


def create_app(request_queue) -> Flask:
    """
    Create and configure a Flask app for image prediction.

    This function initializes a Flask app, sets up necessary configurations,
    starts image preparation and batch prediction processes, and returns the
    configured app instance.

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

    app.request_queue = request_queue

    # Register blueprints
    app.register_blueprint(main)

    return app


if __name__ == '__main__':
    # Set up logging
    logger = setup_logging("INFO")

    # Get Loghi-HTR options from environment variables
    logger.info("Getting Loghi-HTR options from environment variables")
    model_path = get_env_variable("LOGHI_MODEL_PATH")
    batch_size = int(get_env_variable("LOGHI_BATCH_SIZE", "256"))
    output_path = get_env_variable("LOGHI_OUTPUT_PATH")
    max_queue_size = int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000"))

    # Get GPU options from environment variables
    logger.info("Getting GPU options from environment variables")
    gpus = get_env_variable("LOGHI_GPUS", "0")

    # Start the processing and prediction processes
    logger.info("Starting processing and prediction processes")
    request_queue, preparation_process, prediction_process = start_processes(
        batch_size,
        max_queue_size,
        model_path,
        output_path,
        gpus
    )

    # Create and run the Flask app
    app = create_app(request_queue)

    app.run(debug=True)
