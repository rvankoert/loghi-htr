# Imports

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from utils.text import Tokenizer


def initialize_data_manager(config: Config,
                            tokenizer: Tokenizer,
                            model: tf.keras.Model,
                            augmentation_model: tf.keras.Model) -> DataManager:
    """
    Initializes a data manager with specified parameters and based on the input
    shape of a given model.

    Parameters
    ----------
    config : Config
        A Config containing various arguments to configure the data manager
        (e.g., batch size, image size, lists for training, validation, etc.).
    tokenizer : Tokenizer
        The tokenizer to be used for tokenizing text.
    model : tf.keras.Model
        The Keras model, used to derive input dimensions for the data manager.
    augmentation_model : tf.keras.Model
        The Keras model, used to derive input dimensions for the data manager.

    Returns
    -------
    DataManager
        An instance of DataManager configured as per the provided arguments and
        model.
    """

    model_height = model.layers[0].output.shape[2]
    model_channels = model.layers[0].output.shape[3]
    img_size = (model_height, config["width"], model_channels)

    return DataManager(
        img_size=img_size,
        config=config,
        tokenizer=tokenizer,
        augmentation_model=augmentation_model
    )
