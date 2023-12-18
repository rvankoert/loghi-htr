# Imports

# > Standard library
import datetime
import logging
from multiprocessing.queues import Full

# > Local dependencies
from app_utils import extract_request_data

# > Third party dependencies
import flask
from flask import Blueprint, jsonify, current_app as app
from prometheus_client import generate_latest


main = Blueprint('main', __name__)


@main.route('/predict', methods=['POST'])
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
    image_file, group_id, identifier, model = extract_request_data()

    logger = logging.getLogger(__name__)

    logger.debug(f"Data received: {group_id}, {identifier}")
    logger.debug(f"Adding {identifier} to queue")
    logger.debug(f"Using model {model}")

    try:
        app.request_queue.put((image_file, group_id, identifier, model),
                              block=True, timeout=15)
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
def prometheus() -> bytes:
    """
    Endpoint for getting prometheus statistics
    """
    return generate_latest()
