# Imports

# > Standard library
import logging
import os
from typing import List, Tuple

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.loader import DataLoader
from setup.config import Config


def initialize_data_loader(config: Config,
                           charlist: List[str],
                           model: tf.keras.Model) -> DataLoader:
    """
    Initializes a data loader with specified parameters and based on the input
    shape of a given model.

    Parameters
    ----------
    config : Config
        A Config containing various arguments to configure the data loader
        (e.g., batch size, image size, lists for training, validation, etc.).
    charlist : List[str]
        A list of characters to be used by the data loader.
    model : tf.keras.Model
        The Keras model, used to derive input dimensions for the data loader.

    Returns
    -------
    DataLoader
        An instance of DataLoader configured as per the provided arguments and
        model.

    Notes
    -----
    The DataLoader is initialized with parameters like image size, batch size,
    and various data augmentation options. These parameters are derived from
    both the `args` namespace and the input shape of the provided `model`.
    """

    model_height = model.layers[0].input_shape[0][2]
    model_channels = model.layers[0].input_shape[0][3]
    img_size = (model_height, config["width"], model_channels)

    return DataLoader(
        batch_size=config["batch_size"],
        img_size=img_size,
        train_list=config["train_list"],
        test_list=config["test_list"],
        validation_list=config["validation_list"],
        inference_list=config["inference_list"],
        char_list=charlist,
        do_binarize_sauvola=config["do_binarize_sauvola"],
        do_binarize_otsu=config["do_binarize_otsu"],
        multiply=config["multiply"],
        elastic_transform=config["elastic_transform"],
        random_crop=config["random_crop"],
        random_width=config["random_width"],
        check_missing_files=config["check_missing_files"],
        distort_jpeg=config["distort_jpeg"],
        replace_final_layer=config["replace_final_layer"],
        normalization_file=config["normalization_file"],
        use_mask=config["use_mask"],
        do_random_shear=config["do_random_shear"],
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
            raise FileNotFoundError(
                "Charlist not found at: %s and replace_final_layer is False.",
                charlist_location)

        logging.info("Using charlist: %s", charlist)
        logging.info("Charlist length: %s", len(charlist))

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
