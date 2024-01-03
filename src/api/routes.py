# Imports

# > Standard library
import datetime
import logging
from multiprocessing.queues import Full

# > Local dependencies
from app_utils import extract_request_data
from simple_security import session_key_required

# > Third party dependencies
import flask
from flask import Blueprint, jsonify, current_app as app
from prometheus_client import generate_latest


logger = logging.getLogger(__name__)
main = Blueprint('main', __name__)


@main.route('/predict', methods=['POST'])
@session_key_required
def predict() -> flask.Response:
    """
    Endpoint to receive image data and queue it for prediction.

    Receives a POST request containing an image, group_id, and identifier.
    The data is then queued for further processing and prediction.

    Expected POST data
    ------------------
    image : file
        The image file to be processed.
    group_id : str
        The group ID associated with the image.
    identifier : str
        An identifier for the image.

    Returns
    -------
    Flask.Response
        A JSON response containing a status message, timestamp, group_id,
        and identifier. The HTTP status code is 202 (Accepted).

    Side Effects
    ------------
    - Logs debug messages regarding the received data and queuing status.
    - Adds the received data to the global request queue.
    """

    # Add incoming request to queue
    # Here, we're just queuing the raw data.
    image_file, group_id, identifier, model, whitelist = extract_request_data()

    logger.debug(f"Data received: {group_id}, {identifier}")
    logger.debug(f"Adding {identifier} to queue")
    logger.debug(f"Using model {model}")
    logger.debug(f"Using whitelist {whitelist}")

    try:
        app.request_queue.put((image_file, group_id, identifier,
                               model, whitelist), block=True, timeout=15)
    except Full:
        response = jsonify({
            "status": "error",
            "code": 429,
            "message": "The server is currently processing a high volume of "
                       "requests. Please try again later.",
            "timestamp": datetime.datetime.now().isoformat(),
            "group_id": group_id,
            "identifier": identifier,
        })

        response.status_code = 429

        logger.warning("Request queue is full. Maybe one of the workers has "
                       "died?")

        return response

    response = jsonify({
        "status": "Request received",
        "code": 202,
        "message": "Your request is being processed",
        "timestamp": datetime.datetime.now().isoformat(),
        "group_id": group_id,
        "identifier": identifier,
    })

    response.status_code = 202

    return response


@main.route("/prometheus", methods=["GET"])
@session_key_required
def prometheus() -> bytes:
    """
    Endpoint for getting prometheus statistics
    """
    return generate_latest()


@main.route("/health", methods=["GET"])
@session_key_required
def health() -> flask.Response:
    """
    Endpoint for getting health status
    """

    for name, worker in app.workers.items():
        if not worker.is_alive():
            logger.error(f"{name} worker is not alive")
            response = jsonify({
                "status": "unhealthy",
                "code": 500,
                "message": f"{name} worker is not alive",
                "timestamp": datetime.datetime.now().isoformat()
            })
            response.status_code = 500

            return response

    response = jsonify({
        "status": "healthy",
        "code": 200,
        "message": "All workers are alive",
        "timestamp": datetime.datetime.now().isoformat()
    })
    response.status_code = 200

    return response
