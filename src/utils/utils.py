# Imports

# > Standard Library
import os
from typing import Optional, Dict, Any

# > Third party libraries
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import sparse_tensor, dtypes
from tensorflow.python.ops import sparse_ops, array_ops, math_ops
from tensorflow.python.ops import ctc_ops as ctc
from numpy import exp


class Utils:
    def __init__(self, chars, use_mask):
        self.set_charlist(chars=chars, use_mask=use_mask)

    def softmax(self, vector):
        e = exp(vector)
        return e / e.sum()

    def set_charlist(self, chars, use_mask=False, num_oov_indices=0):
        self.charList = chars
        if num_oov_indices > 0:
            self.charList.insert(1, '[UNK]')
        if not self.charList:
            raise Exception('No characters found in character list')
        if use_mask:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token='', oov_token='[UNK]',
                encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token='',
                encoding="UTF-8",
                invert=True
            )
        else:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token=None, oov_token='[UNK]',
                encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None,
                encoding="UTF-8",
                invert=True
            )


def shape(x):
    """Returns the symbolic shape of a tensor or variable.

    Args:
        x: A tensor or variable.

    Returns:
        A symbolic shape (which is itself a tensor).

    Examples:

    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = tf.keras.backend.variable(value=val)
    >>> tf.keras.backend.shape(kvar)
    <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>
    >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
    >>> tf.keras.backend.shape(input)
    <KerasTensor: shape=(3,) dtype=int32 inferred_value=[2, 4, 5] ...>

    """
    return array_ops.shape(x)


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100):
    """
    Decodes the prediction using CTC Decoder.

    Parameters
    ----------
    y_pred : ndarray
        Predicted probabilities.
    input_length : ndarray
        Length of the input sequences.
    greedy : bool, optional
        If true, use greedy decoder, else use beam search decoder.
    beam_width : int, optional
        Width of the beam for beam search decoder.

    Returns
    -------
    list of ndarray
        Decoded sequences.
    ndarray
        Log probabilities of the decoded sequences.
    """

    num_samples, num_steps = y_pred.shape[:2]
    y_pred = tf.math.log(tf.transpose(
        y_pred, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length, merge_repeated=True)
    else:
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=y_pred, sequence_length=input_length,
            beam_width=beam_width, top_paths=1)

    decoded_dense = [tf.sparse.to_dense(
        st, default_value=-1) for st in decoded]
    return decoded_dense, log_prob


def decode_batch_predictions(pred, utils, greedy=True,
                             beam_width=1, num_oov_indices=0):
    """
    Decodes batch predictions using CTC Decoder.

    Parameters
    ----------
    pred : ndarray
        Predicted probabilities from the model.
    utils : object
        Utility object for character conversion.
    greedy : bool, optional
        If true, use greedy decoder, else use beam search decoder.
    beam_width : int, optional
        Width of the beam for beam search decoder.
    num_oov_indices : int, optional
        Number of out-of-vocabulary indices.

    Returns
    -------
    list of tuple
        List of tuples containing confidence and decoded text.
    """

    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    pred = tf.cast(pred, tf.float32)
    ctc_decoded, log_probs = ctc_decode(pred, input_length=input_len,
                                        greedy=greedy, beam_width=beam_width)

    output_texts = []
    for i, decoded_array in enumerate(ctc_decoded[0]):
        decoded_array += num_oov_indices
        chars = utils.num_to_char(decoded_array)
        text = tf.strings.reduce_join(chars).numpy().decode("utf-8")

        # Calculate the effective steps for each sample in the batch
        # That is before the first blank character
        # Find indices of the first occurrence of -1 in each sequence
        time_steps = np.array(decoded_array == -1).argmax(axis=0)
        time_steps = time_steps if time_steps > 0 else len(decoded_array)

        # Normalize the confidence score based on the number of timesteps
        confidence = np.exp(-log_probs[i][0] / time_steps
                            if greedy else log_probs[i][0] / time_steps)

        output_texts.append((confidence, text))

    return output_texts


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
