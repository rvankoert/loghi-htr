# Imports

# > Standard Library
import os
from pathlib import Path
import sys
from typing import Tuple

# > Third party libraries
import tensorflow as tf
import numpy as np

# > Local dependencies
from vis_arg_parser import get_args

# Add the above directory to the path
sys.path.append(str(Path(__file__).resolve().parents[1] / '../src'))  # noqa: E402

width_padding = 50


def preprocess_image(image_path: str, model_channels: int, target_height: int) \
        -> Tuple[np.ndarray, tf.Tensor, tf.Tensor]:
    """
    Prepare an image for input to a model.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    model_channels : int
        The number of channels expected by the model.
    target_height : int
        The image target height

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - img : numpy.ndarray
            The preprocessed image ready for input to the model.
        - image_width : tf.Tensor
            The width of the preprocessed image.
        - image_height : tf.Tensor
            The height of the preprocessed image.

    Notes
    -----
    This function reads an image from the specified path, resizes and
    normalizes it, pads the image, and prepares it for input to a model.

    Examples
    --------
    >>> image_path = "/path/to/your/image.jpg"
    >>> model_channels = 3
    >>> img, width, height = preprocess_image(image_path, model_channels)
    # Returns a tuple with the preprocessed image and its dimensions.
    """
    # Remake data_generator parts
    original_image = tf.io.read_file(image_path)
    original_image = tf.image.decode_image(
        original_image, channels=model_channels)
    original_image = tf.image.resize(original_image,
                                     [target_height, 99999],
                                     preserve_aspect_ratio=True)

    image_width = tf.shape(original_image)[1]
    image_height = tf.shape(original_image)[0]

    # Normalize the image and something else
    # Pad the image
    img = 0.5 - (original_image / 255.0)
    img = tf.image.resize_with_pad(img, image_height, image_width + width_padding)
    img = tf.transpose(img, perm=[1, 0, 2])
    img = np.expand_dims(img, axis=0)
    return img, image_width, image_height


def init_pre_trained_model():
    """
    Initialize a pre-trained model for the visualizer.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - model : tf.keras.Model
            The loaded pre-trained model.
        - model_channels : int
            The number of channels expected by the model.
        - model_path : str
            The path to the loaded pre-trained model.

    Notes
    -----
    This function initializes a pre-trained model, loading it from the
    specified path. It also sets seeds for reproducibility and provides
    model-related information.

    Examples
    --------
    >>> loaded_model, channels, path, model_height = init_pre_trained_model()
    # Returns a tuple with the loaded model, number of channels, and the model
    path.
    """
    # Retrieve args and load model
    args = get_args()

    if args.existing_model:
        if not os.path.exists(args.existing_model):
            raise FileNotFoundError("Please provide a valid path to an "
                                    "existing model, you provided: "
                                    + str(args.existing_model))
        model_path = args.existing_model
    else:
        raise ValueError("Please provide a path to an existing model")

    seed = args.seed
    # Set seed for plots to check changes in preprocessing
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if any(file.endswith(".keras") for file in os.listdir(model_path)):
        # Look for a .keras file
        model_file = next((os.path.join(model_path, file) for file in os.listdir(
            model_path) if file.endswith(".keras")), None)
    else:
        # Else take .pb file approach
        model_file = model_path

    # Load model and custom objects
    model = tf.keras.saving.load_model(model_file,
                                       compile=False)

    model_channels = model.input_shape[3]
    model_height = model.input_shape[2]
    model.summary()

    return model, model_channels, model_path, model_height
