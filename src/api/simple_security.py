import functools
import uuid

import flask
from flask import request, Response, jsonify, Flask
import json
import logging


class SimpleSecurity:
    def __init__(self, app: Flask, config: dict):
        if not app or not config:
            raise ValueError("App and config must be provided")

        app.extensions["security"] = self
        self.app = app
        self.config = config
        self.enabled = self._str_to_bool(config.get("enabled", "false"))

        # Load API key user only if security is enabled
        if self.enabled:
            self.api_key_user = self._load_api_key_user(
                config.get("key_user_json"))
            self.session_key_user = {}
            self._register_login_resource()
        else:
            self.api_key_user = {}
            self.session_key_user = {}

    @staticmethod
    def _str_to_bool(s):
        """Convert a string to a boolean."""
        return s.lower() in ["true", "1", "yes", "t"]

    def _load_api_key_user(self, key_user_json: str):
        if not key_user_json:
            raise ValueError(
                "key_user_json must be provided when security is enabled")

        try:
            return json.loads(key_user_json)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {e}")
            raise ValueError("Invalid key_user_json format")

    def is_known_session_key(self, session_key: str):
        return session_key in self.session_key_user

    def _register_login_resource(self):
        @self.app.route("/login", methods=["POST"])
        def login():
            api_key = request.headers.get("Authorization")
            if api_key:
                session_key = self._login(api_key)
                if session_key:
                    return Response(status=204,
                                    headers={"X_AUTH_TOKEN": session_key})
            return jsonify(message="Unauthorized"), 401

    def _login(self, api_key: str) -> str:
        if self.enabled and api_key in self.api_key_user:
            session_key = str(uuid.uuid4())
            self.session_key_user[session_key] = self.api_key_user[api_key]
            return session_key


def session_key_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs) -> Response:
        if not (security_ := flask.current_app.extensions.get("security")):
            return func(*args, **kwargs)

        session_key = request.headers.get("Authorization")
        if security_.enabled and security_.is_known_session_key(session_key):
            return func(*args, **kwargs)

        return jsonify(message="Expected a valid session key"), 401

    return decorator
