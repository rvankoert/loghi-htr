# Imports

# > Standard library
import logging
import os
from typing import Tuple

# > Third-party dependencies
from flask import request


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging with the specified level and return a logger instance.

    Parameters
    ----------
    level : str, optional
        Desired logging level. Supported values are "DEBUG", "INFO",
        "WARNING", "ERROR". Default is "INFO".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    # Set up the basic logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging_levels[level],
    )

    return logging.getLogger(__name__)


def extract_request_data() -> Tuple[bytes, str, str]:
    """
    Extract image and other form data from the current request.

    Returns
    -------
    tuple of (bytes, str, str)
        image_content : bytes
            Content of the uploaded image.
        group_id : str
            ID of the group from form data.
        identifier : str
            Identifier from form data.

    Raises
    ------
    ValueError
        If required data (image, group_id, identifier) is missing or if the
        image format is invalid.
    """

    # Extract the uploaded image
    image_file = request.files.get('image')
    if not image_file:
        raise ValueError("No image provided.")

    # Validate image format
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or image_file.filename.rsplit('.', 1)[1]\
            .lower() not in allowed_extensions:
        raise ValueError(
            "Invalid image format. Allowed formats: png, jpg, jpeg, gif")

    image_content = image_file.read()

    # Extract other form data
    group_id = request.form.get('group_id')
    if not group_id:
        raise ValueError("No group_id provided.")

    identifier = request.form.get('identifier')
    if not identifier:
        raise ValueError("No identifier provided.")

    return image_content, group_id, identifier


def get_env_variable(var_name: str, default_value: str = None) -> str:
    """
    Retrieve an environment variable's value or use a default value.

    Parameters
    ----------
    var_name : str
        The name of the environment variable.
    default_value : str, optional
        Default value to use if the environment variable is not set.
        Default is None.

    Returns
    -------
    str
        Value of the environment variable or the default value.

    Raises
    ------
    ValueError
        If the environment variable is not set and no default value is
        provided.
    """

    logger = logging.getLogger(__name__)

    value = os.environ.get(var_name)
    if value is None:
        if default_value is None:
            raise ValueError(
                f"Environment variable {var_name} not set and no default "
                "value provided.")
        logger.warning(
            f"Environment variable {var_name} not set. Using default value: "
            f"{default_value}")
        return default_value

    logger.debug(f"Environment variable {var_name} set to {value}")
    return value
