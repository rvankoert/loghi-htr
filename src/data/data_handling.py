# Imports

# > Standard library
import logging
import os
from typing import List, Tuple

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config


def initialize_data_manager(config: Config,
                            charlist: List[str],
                            model: tf.keras.Model,
                            augment_model: tf.keras.Sequential) -> DataManager:
    """
    Initializes a data manager with specified parameters and based on the input
    shape of a given model.

    Parameters
    ----------
    config : Config
        A Config containing various arguments to configure the data manager
        (e.g., batch size, image size, lists for training, validation, etc.).
    charlist : List[str]
        A list of characters to be used by the data manager.
    model : tf.keras.Model
        The Keras model, used to derive input dimensions for the data manager.
    augment_model : tf.keras.Sequential
        The Keras model used for data augmentation.

    Returns
    -------
    DataManager
        An instance of DataManager configured as per the provided arguments and
        model.
    """

    model_height = model.layers[0].input_shape[0][2]
    model_channels = model.layers[0].input_shape[0][3]
    img_size = (model_height, config["width"], model_channels)

    return DataManager(
        img_size=img_size,
        charlist=charlist,
        augment_model=augment_model,
        config=config
    )


def load_initial_charlist(charlist_location: str, existing_model: str,
                          output_directory: str, replace_final_layer: bool) \
        -> Tuple[List[str], bool]:
    """
    Loads the initial character list from the specified location or model
    directory.

    Parameters
    ----------
    charlist_location : str
        The location where the character list is stored.
    existing_model : str
        The path to the existing model, which might contain the character list.
    output_directory : str
        The directory where output files are stored, which might contain the
        character list.
    replace_final_layer : bool
        A flag indicating whether the final layer of the model is being
        replaced.

    Returns
    -------
    Tuple[List[str], bool]
        A tuple containing the character list and a flag indicating whether
        padding was removed from the character list.

    Raises
    ------
    FileNotFoundError
        If the character list file is not found and the final layer is not
        being replaced.

    Notes
    -----
    The function first determines the location of the character list file based
    on the provided paths and the `replace_final_layer` flag. It then loads the
    character list from the file if it exists.
    """

    # Set the character list location
    if not charlist_location and existing_model:
        charlist_location = existing_model + '/charlist.txt'
    elif not charlist_location:
        charlist_location = output_directory + '/charlist.txt'

    # Load the character list
    charlist = []
    removed_padding = False

    # We don't need to load the charlist if we are replacing the final layer
    if not replace_final_layer:
        if os.path.exists(charlist_location):
            with open(charlist_location, encoding="utf-8") as file:
                for char in file.read():
                    if char == '':
                        logging.warning("Found padding character in the "
                                        "charlist. Removing it.")
                        removed_padding = True
                    else:
                        charlist.append(char)
            logging.info("Using charlist from: %s", charlist_location)
        else:
            raise FileNotFoundError("Charlist not found at: "
                                    f"{charlist_location} and "
                                    "replace_final_layer is False.")

    return charlist, removed_padding


def save_charlist(charlist: List[str],
                  output: str) -> None:
    """
    Saves the given character list to a specified location.

    Parameters
    ----------
    charlist : List[str]
        The character list to be saved.
    output : str
        The base output directory where the character list file is to be saved.

    Notes
    -----
    This function saves the provided character list to a file, either at a
    specified location or by default in the output directory under the filename
    'charlist.txt'.
    """

    # Save the new charlist
    with open(f"{output}/charlist.txt", 'w', encoding="utf-8") as chars_file:
        chars_file.write(str().join(charlist))
