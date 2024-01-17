# Imports

# > Third-party dependencies
import numpy as np
import tensorflow as tf

# > Local imports
from utils.text import Tokenizer


def ctc_decode(y_pred: np.ndarray, input_length: np.ndarray,
               greedy: bool = True, beam_width: int = 100) -> tuple:
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

    # Get the number of samples and timesteps
    num_samples, num_steps = y_pred.shape[:2]

    # Transpose the predictions
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2])
                         + tf.keras.backend.epsilon())

    # Convert the input length to int32
    input_length = tf.cast(input_length, tf.int32)

    # Decode the sequence
    if greedy or beam_width == 1:
        decoded, log_prob = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length, merge_repeated=True)
    else:
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=y_pred, sequence_length=input_length,
            beam_width=beam_width, top_paths=1)

    # Convert sparse to dense
    decoded_dense = [tf.sparse.to_dense(st, default_value=-1)
                     for st in decoded]

    return decoded_dense, log_prob


def decode_batch_predictions(pred: np.ndarray, tokenizer: Tokenizer,
                             greedy: bool = True, beam_width: int = 1,
                             num_oov_indices: int = 1) -> list:
    """
    Decodes batch predictions using CTC Decoder.

    Parameters
    ----------
    pred : ndarray
        Predicted probabilities from the model.
    tokenizer : Tokenizer
        Tokenizer object containing character list and other information.
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

    # Get the input length for each sample
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    # Convert the predictions to float32
    pred = tf.cast(pred, tf.float32)

    # Decode the predictions
    ctc_decoded, log_probs = ctc_decode(pred, input_length=input_len,
                                        greedy=greedy, beam_width=beam_width)

    # Convert the decoded sequence to text
    output_texts = []
    for i, decoded_array in enumerate(ctc_decoded[0]):
        decoded_array += num_oov_indices

        # Calculate the effective steps for each sample in the batch
        # That is before the first blank character
        time_steps = np.array(decoded_array == num_oov_indices-1)\
            .argmax(axis=0)
        time_steps = time_steps if time_steps > 0 else len(decoded_array)

        # Normalize the confidence score based on the number of timesteps
        text = tokenizer.decode(decoded_array).strip().replace("", "")
        confidence = np.exp(-log_probs[i][0] / time_steps
                            if greedy else log_probs[i][0] / time_steps)

        output_texts.append((confidence, text))

    return output_texts
