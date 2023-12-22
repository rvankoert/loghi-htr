# Imports

# > Standard library

# > Local dependencies
import errors
from routes import main
from app_utils import setup_logging, get_env_variable, start_workers
from simple_security import SimpleSecurity

# > Third-party dependencies
from flask import Flask


def create_app() -> Flask:
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
    - Initializes and starts preparation, prediction, and decoding processes.
    - Logs various messages regarding the app and process initialization.
    """

    # Set up logging
    logging_level = get_env_variable("LOGGING_LEVEL", "INFO")
    logger = setup_logging(logging_level)

    # Get Loghi-HTR options from environment variables
    logger.info("Getting Loghi-HTR options from environment variables")
    batch_size = int(get_env_variable("LOGHI_BATCH_SIZE", "256"))
    model_path = get_env_variable("LOGHI_MODEL_PATH")
    output_path = get_env_variable("LOGHI_OUTPUT_PATH")
    max_queue_size = int(get_env_variable("LOGHI_MAX_QUEUE_SIZE", "10000"))
    patience = float(get_env_variable("LOGHI_PATIENCE", "0.5"))

    # Get GPU options from environment variables
    logger.info("Getting GPU options from environment variables")
    gpus = get_env_variable("LOGHI_GPUS", "0")

    # Create Flask app
    logger.info("Creating Flask app")
    app = Flask(__name__)

    # Register error handler
    app.register_error_handler(ValueError, errors.handle_invalid_usage)
    app.register_error_handler(405, errors.method_not_allowed)

    # Add security to app
    enable_security = get_env_variable("SECURITY_ENABLED", False)
    logger.info(f"Security enabled: {enable_security}")

    # Get API key user JSON string from environment variable
    api_key_user_json_string = get_env_variable("API_KEY_USER_JSON_STRING", "")
    SimpleSecurity(app, enable_security, api_key_user_json_string)

    # Start the worker processes
    logger.info("Starting worker processes")
    workers, queues = start_workers(batch_size, max_queue_size, output_path,
                                    gpus, model_path, patience)

    # Add request queue to the app
    app.request_queue = queues["Request"]

    # Add the workers to the app
    app.workers = workers

    # Register blueprints
    app.register_blueprint(main)

    return app
