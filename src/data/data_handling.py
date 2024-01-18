# Imports

# > Standard library
import argparse
import logging
import os
from typing import List, Optional

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from data.loader import DataLoader


def initialize_data_loader(args: argparse.Namespace, char_list: List[str],
                           model: tf.keras.Model) -> DataLoader:
    """
    Initializes a data loader with specified parameters and based on the input
    shape of a given model.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing various arguments to configure the data loader
        (e.g., batch size, image size, lists for training, validation, etc.).
    char_list : List[str]
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
    img_size = (model_height, args.width, model_channels)

    return DataLoader(
        batch_size=args.batch_size,
        img_size=img_size,
        train_list=args.train_list,
        validation_list=args.validation_list,
        test_list=args.test_list,
        inference_list=args.inference_list,
        char_list=char_list,
        aug_binarize_sauvola=args.aug_binarize_sauvola,
        aug_binarize_otsu=args.aug_binarize_otsu,
        multiply=args.multiply,
        augment=args.augment,
        aug_elastic_transform=args.aug_elastic_transform,
        aug_random_crop=args.aug_random_crop,
        aug_random_width=args.aug_random_width,
        check_missing_files=args.check_missing_files,
        aug_distort_jpeg=args.aug_distort_jpeg,
        replace_final_layer=args.replace_final_layer,
        normalization_file=args.normalization_file,
        use_mask=args.use_mask,
        aug_random_shear=args.aug_random_shear
    )


def load_initial_charlist(charlist_location: str, existing_model: str,
                          output_directory: str, replace_final_layer: bool) \
        -> List[str]:
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
    List[str]
        A list of characters loaded from the character list file.

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
    char_list = []

    # We don't need to load the charlist if we are replacing the final layer
    if not replace_final_layer:
        if os.path.exists(charlist_location):
            with open(charlist_location) as file:
                char_list = [char for char in file.read()]
            logging.info(f"Using charlist from: {charlist_location}")
        else:
            raise FileNotFoundError(
                f"Charlist not found at: {charlist_location} and "
                "replace_final_layer is False. Exiting...")

        logging.info(f"Using charlist: {char_list}")
        logging.info(f"Charlist length: {len(char_list)}")

    return char_list


def save_charlist(charlist: List[str], output: str,
                  output_charlist_location: Optional[str] = None) -> None:
    """
    Saves the given character list to a specified location.

    Parameters
    ----------
    charlist : List[str]
        The character list to be saved.
    output : str
        The base output directory where the character list file is to be saved.
    output_charlist_location : Optional[str]
        The specific location where the character list file is to be saved. If
        not provided, it defaults to a location within the output directory.

    Notes
    -----
    This function saves the provided character list to a file, either at a
    specified location or by default in the output directory under the filename
    'charlist.txt'.
    """

    # Save the new charlist
    if not output_charlist_location:
        output_charlist_location = output + '/charlist.txt'
    with open(output_charlist_location, 'w') as chars_file:
        chars_file.write(str().join(charlist))