import os
import logging
from typing import List, Optional

from fastapi import UploadFile, Form, File, HTTPException


def get_env_variable(var_name: str, default_value: Optional[str] = None) -> str:
    """
    Retrieve an environment variable's value or use a default value.
    If default_value is not provided and variable is not set, raises ValueError.
    """
    logger = logging.getLogger(__name__)  # Use specific logger for this utility
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
    Extract image and other form data from the current request.
    """
    # Validate image format
    allowed_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = image.filename.split(".")[-1].lower() if image.filename else ""
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: "
            f"{', '.join(allowed_extensions)}",
        )

    # Read image content
    image_content = await image.read()

    # Check if the image content is empty
    if not image_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty. Please upload a valid image file.",
        )

    return image_content, group_id, identifier, model, whitelist
