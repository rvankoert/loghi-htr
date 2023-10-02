# Imports

# > Standard library
import datetime

# > Third-party dependencies
import flask
from flask import jsonify, current_app as app


def handle_invalid_usage(error: Exception) -> flask.Response:
    """
    Handle invalid request data by sending a 400 Bad Request response.

    This function constructs a JSON response detailing the error and logs the
    error on the server side.

    Parameters
    ----------
    error : Exception
        The caught exception detailing the invalid usage.

    Returns
    -------
    Flask.Response
        A Flask response object with a 400 Bad Request status and a JSON body
        detailing the error.

    Side Effects
    ------------
    - Logs an error message on the server side.
    """

    # Log the error server-side
    response = jsonify({
        "status": "error",
        "code": 400,
        "message": "Invalid request data.",
        "details": str(error),
        "timestamp": datetime.datetime.now().isoformat()
    })

    app.logger.error(f"Invalid request data: {error}")
    response.status_code = 400  # Set the status code to 400 Bad Request

    return response


def method_not_allowed(error) -> flask.Response:
    """
    Handle invalid request methods by sending a 405 Method Not Allowed
    response.

    This function constructs a JSON response detailing the error and logs the
    error on the server side.

    Parameters
    ----------
    error : Exception
        The caught exception detailing the invalid usage.

    Returns
    -------
    Flask.Response
        A Flask response object with a 405 Method Not Allowed status and a JSON
        body detailing the error.
    """

    response = jsonify({
        "status": "Error",
        "code": 405,
        "message": "Method not allowed. Please use the appropriate request "
                   "method.",
        "details": str(error),
        "timestamp": datetime.datetime.now().isoformat()
    })

    app.logger.error(error)
    response.status_code = 405

    return response
