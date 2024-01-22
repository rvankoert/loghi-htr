# Imports

# > Standard library
import datetime
import functools
import json
import logging
import uuid

# > Third-party dependencies
import flask
from flask import request, Response, jsonify, Flask


class SimpleSecurity:
    """
    A simple security class that provides basic API key and session key
    management for a Flask application.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    config : dict
        A configuration dictionary containing security settings.

    Raises
    ------
    ValueError
        If either `app` or `config` is not provided.

    Attributes
    ----------
    app : Flask
        The Flask application instance.
    config : dict
        The configuration dictionary.
    enabled : bool
        Indicates whether security is enabled.
    api_key_user : dict
        A dictionary mapping API keys to user data.
    session_key_user : dict
        A dictionary mapping session keys to user data.
    """

    def __init__(self, app: Flask, config: dict):
        if not app or not config:
            raise ValueError("App and config must be provided")

        app.extensions["security"] = self
        self.app = app
        self.config = config
        self.enabled = self._security_enabled(config.get("enabled", "false"))

        # Load API key user only if security is enabled
        if self.enabled:
            self.api_key_user = self._load_api_key_user(
                config.get("key_user_json"))
            self.session_key_user = {}
            self._register_login_resource()
        else:
            self.api_key_user = {}
            self.session_key_user = {}

    def _security_enabled(self, s: str) -> bool:
        """
        Convert a string value to a boolean.

        Parameters
        ----------
        s : str
            A string representing a boolean value.

        Returns
        -------
        bool
            True if the string represents a positive boolean value, False
            otherwise.
        """
        return s.lower() in ["true", "1", "yes", "t"]

    def _load_api_key_user(self, key_user_json: str) -> dict:
        """
        Load API key user data from a JSON string.

        Parameters
        ----------
        key_user_json : str
            A JSON string representing API key user data.

        Returns
        -------
        dict
            A dictionary mapping API keys to user data.

        Raises
        ------
        ValueError
            If `key_user_json` is not provided or contains invalid JSON.
        """

        if not key_user_json:
            raise ValueError(
                "key_user_json must be provided when security is enabled")

        try:
            return json.loads(key_user_json)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {e}")
            raise ValueError("Invalid key_user_json format")

    def is_known_session_key(self, session_key: str) -> bool:
        """
        Check if a session key is known (i.e., exists in the session_key_user
        dictionary).

        Parameters
        ----------
        session_key : str
            The session key to check.

        Returns
        -------
        bool
            True if the session key is known, False otherwise.
        """
        return session_key in self.session_key_user

    def _register_login_resource(self):
        """
        Register a login resource on the Flask application to handle login
        requests.
        """
        @self.app.route("/login", methods=["POST"])
        def login():
            # Check if an API key is provided in the request headers
            api_key = request.headers.get("Authorization")
            if api_key:

                # Attempt to login
                session_key = self._login(api_key)
                if session_key:
                    response = jsonify({"status": "success",
                                        "code": 204,
                                        "message": "Login successful",
                                        "timestamp":
                                        datetime.datetime.now().isoformat()})
                    response.status_code = 204
                    response.headers["X_AUTH_TOKEN"] = session_key
                    return response

            # Either no API key was provided or login failed
            response = jsonify({"status": "unauthorized",
                                "code": 401,
                                "message": "Expected a valid API key",
                                "timestamp":
                                datetime.datetime.now().isoformat()})
            response.status_code = 401
            return response

    def _login(self, api_key: str) -> str:
        """
        Handle the login process using an API key.

        Parameters
        ----------
        api_key : str
            The API key used for login.

        Returns
        -------
        str
            A new session key if login is successful, or an empty string
            otherwise.
        """

        if self.enabled and api_key in self.api_key_user:
            session_key = str(uuid.uuid4())
            self.session_key_user[session_key] = self.api_key_user[api_key]
            return session_key


def session_key_required(func):
    """
    A decorator that checks for a valid session key before allowing access to a
    view function.

    Parameters
    ----------
    func : Callable
        The view function to be decorated.

    Returns
    -------
    Callable
        The decorated function.

    Raises
    ------
    Unauthorized
        If a valid session key is not provided in the request headers.
    """

    @functools.wraps(func)
    def decorator(*args, **kwargs) -> Response:
        security_ = flask.current_app.extensions.get("security")
        if not security_ or (security_ and not security_.enabled):
            return func(*args, **kwargs)

        session_key = request.headers.get("Authorization")
        if security_.enabled and security_.is_known_session_key(session_key):
            return func(*args, **kwargs)

        response = jsonify({"status": "unauthorized",
                            "code": 401,
                            "message": "Expected a valid session key",
                            "timestamp": datetime.datetime.now().isoformat()})
        response.status_code = 401
        return response

    return decorator
