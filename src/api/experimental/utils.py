# Imports

# > Standard Library
import logging
import os
from typing import List, Optional

# > Third-party Dependencies
from fastapi import File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)


def get_env_variable(var_name: str, default_value: Optional[str] = None) -> str:
    """
    Retrieve an environment variable's value or return a default.

    Parameters
    ----------
    var_name : str
        Name of the environment variable to retrieve.
    default_value : Optional[str], optional
        Fallback value if the variable is not set. If None and the variable is
        unset, raises a ValueError.

    Returns
    -------
    str
        Value of the environment variable or the provided default.

    Raises
    ------
    ValueError
        If the variable is not set and no default is provided.
    """
    value = os.environ.get(var_name)
    if value is None:
        if default_value is None:
            error_msg = (
                f"Environment variable {var_name} not set and no default "
                "value provided."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.warning(
            "Environment variable %s not set. Using default value: %s",
            var_name,
            default_value,
        )
        return default_value
    logger.debug("Environment variable %s set to %s", var_name, value)
    return value


async def extract_request_data(
    image: UploadFile = File(...),
    group_id: str = Form(...),
    identifier: str = Form(...),
    model: Optional[str] = Form(None),
    whitelist: List[str] = Form([]),
) -> tuple[bytes, str, str, Optional[str], list]:
    """
    Extract and validate image and metadata from a multipart/form-data request.

    Parameters
    ----------
    image : UploadFile
        Uploaded image file from the request.
    group_id : str
        Group ID associated with the prediction request.
    identifier : str
        Unique identifier for the document/image.
    model : Optional[str], optional
        Optional model name to override default model.
    whitelist : List[str], optional
        Optional list of annotations to include in the prediction.

    Returns
    -------
    tuple
        Tuple containing the image content, group ID, identifier, model, and whitelist.

    Raises
    ------
    HTTPException
        If the image format is unsupported or the image is empty.
    """
    allowed_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = image.filename.split(".")[-1].lower() if image.filename else ""
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: "
            f"{', '.join(allowed_extensions)}",
        )

    image_content = await image.read()

    if not image_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please upload a valid image file.",
        )

    return image_content, group_id, identifier, model, whitelist
