# Imports

# > Standard library
import datetime

# > Local dependencies
from app_utils import extract_request_data

# > Third party dependencies
from flask import Blueprint, jsonify, current_app as app

main = Blueprint('main', __name__)


@main.route('/predict', methods=['POST'])
def predict():
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
    image_file, group_id, identifier = extract_request_data()
    app.logger.debug(f"Data received: {group_id}, {identifier}")

    app.logger.debug(f"Adding {identifier} to queue")
    app.request_queue.put((image_file, group_id, identifier))

    response = jsonify({
        "status": "Request received",
        "message": "Your request is being processed",
        "timestamp": datetime.datetime.now().isoformat(),
        "group_id": group_id,
        "identifier": identifier
    })

    return response, 202
