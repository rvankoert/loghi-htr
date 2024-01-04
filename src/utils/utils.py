# Imports

# > Standard Library
import os
from typing import Optional, Dict, Any

# > Third party libraries
import tensorflow as tf
import numpy as np
from keras.models import Model
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


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """Decodes the output of a softmax.

    Can use either greedy search (also known as best path)
    or a constrained dictionary search.

    Args:
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.

    Returns:
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Each decoded sequence has shape (samples, time_steps).
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    input_shape = shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = math_ops.log(array_ops.transpose(
        y_pred, perm=[1, 0, 2]) + tf.keras.backend.epsilon())
    input_length = math_ops.cast(input_length, dtypes.int32)

    if greedy:
        (decoded, log_prob) = ctc.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length)
    else:
        (decoded, log_prob) = ctc.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
            merge_repeated=False)
    decoded_dense = []
    for st in decoded:
        st = sparse_tensor.SparseTensor(
            st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(
            sparse_ops.sparse_tensor_to_dense(sp_input=st, default_value=-1))
    return decoded_dense, log_prob


def decode_batch_predictions(pred, utils, greedy=True, beam_width=1, num_oov_indices=0):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    pred = tf.dtypes.cast(pred, tf.float32)
    top_paths = 1
    output_texts = []
    ctc_decoded = ctc_decode(pred, input_length=input_len,
                             greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    for top_path in range(0, top_paths):
        results = ctc_decoded[0][top_path][:, :]

        # Iterate over the results and get back the text
        output_text = []
        i = 0
        for res in results:
            log_prob = ctc_decoded[1][i][top_path]
            if greedy:
                confidence = np.exp(-log_prob)
            else:
                confidence = np.exp(log_prob)
            i = i + 1
            res = res + num_oov_indices
            chars = utils.num_to_char(res)
            res = tf.strings.reduce_join(chars).numpy().decode("utf-8")
            output_text.append((confidence, res))
        output_texts.append(output_text)
    return output_texts


def normalize_confidence(confidence, predicted_text):
    if len(predicted_text) > 0:
        # we really want 1/number of timesteps in CTC matrix, but len(predicted_text) is next best for now
        confidence = pow(confidence, (1 / len(predicted_text)))
        if confidence < 0:
            confidence = -confidence
    return confidence


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
