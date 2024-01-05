# Imports

# > Standard Library
import os
from typing import Optional, Dict, Any

# > Third party libraries
import tensorflow as tf


class Tokenizer:
    def __init__(self, chars, use_mask):
        self.set_charlist(chars=chars, use_mask=use_mask)

    def set_charlist(self, chars, use_mask=False, num_oov_indices=0):
        self.charList = chars
        if num_oov_indices > 0:
            self.charList.insert(1, '[UNK]')
        if not self.charList:
            raise Exception('No characters found in character list')
        if use_mask:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList),
                num_oov_indices=num_oov_indices, mask_token='',
                oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                num_oov_indices=0, oov_token='', mask_token='',
                encoding="UTF-8", invert=True
            )
        else:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList),
                num_oov_indices=num_oov_indices, mask_token=None,
                oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                num_oov_indices=0, oov_token='', mask_token=None,
                encoding="UTF-8", invert=True
            )


def load_model_from_directory(directory: str,
                              custom_objects: Optional[Dict[str, Any]] = None,
                              compile: bool = True) -> tf.keras.Model:
    """
    Load a TensorFlow Keras model from a specified directory.

    This function supports loading models in both the SavedModel format (.pb)
    and the Keras format (.keras). It first searches for a .pb file to identify
    a SavedModel. If not found, it looks for a .keras file.

    Parameters
    ----------
    directory : str
        The directory where the model is saved.
    custom_objects : Optional[Dict[str, Any]], optional
        Optional dictionary mapping names (strings) to custom classes or
        functions to be considered during deserialization, by default None.
    compile : bool, optional
        Whether to compile the model after loading, by default True.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found in the specified directory.
    """

    # Check for a .pb file (indicating SavedModel format)
    if any(file.endswith('.pb') for file in os.listdir(directory)):
        return tf.keras.models.load_model(directory,
                                          custom_objects=custom_objects,
                                          compile=compile)

    # Look for a .keras file
    model_file = next((os.path.join(directory, file) for file in os.listdir(
        directory) if file.endswith(".keras")), None)

    if model_file:
        return tf.keras.models.load_model(model_file,
                                          custom_objects=custom_objects,
                                          compile=compile)

    raise FileNotFoundError("No suitable model file found in the directory.")
