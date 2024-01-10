# Imports

# > Third-party dependencies
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops, array_ops, ctc_ops


def ctc_batch_cost(y_true: tf.Tensor, y_pred: tf.Tensor,
                   input_length: tf.Tensor, label_length: tf.Tensor):
    """
    Calculate the CTC loss for each batch element.

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor of shape `(samples, max_string_length)` containing the true
        labels.
    y_pred : tf.Tensor
        A tensor of shape `(samples, time_steps, num_categories)` containing
        the predictions, or output of the softmax.
    input_length : tf.Tensor
        A tensor of shape `(samples, 1)` containing the sequence length for
        each batch item in `y_pred`.
    label_length : tf.Tensor
        A tensor of shape `(samples, 1)` containing the sequence length for
        each batch item in `y_true`.

    Returns
    -------
    tf.Tensor
        A tensor of shape `(samples, 1)` containing the CTC loss for each
        element.

    Notes
    -----
    This function utilizes TensorFlow operations to compute the CTC loss for
    each batch item, considering the provided sequence lengths.
    """

    # Squeeze the label and input length tensors to remove the last dimension
    label_length = tf.cast(array_ops.squeeze(label_length), dtype="int32")
    input_length = tf.cast(array_ops.squeeze(input_length), dtype="int32")
    sparse_labels = tf.cast(K.ctc_label_dense_to_sparse(y_true, label_length),
                            dtype="int32")

    # Apply log transformation to predictions and transpose for CTC loss
    # calculation
    y_pred = math_ops.log(array_ops.transpose(
        y_pred, perm=[1, 0, 2]) + K.epsilon())

    # Compute the CTC loss and expand its dimensions to match the required
    # output shape
    return array_ops.expand_dims(
        ctc_ops.ctc_loss(
            inputs=y_pred,
            labels=sparse_labels,
            sequence_length=input_length,
            ignore_longer_outputs_than_inputs=True),
        1)


def CTCLoss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the CTC loss for a batch of data.

    Parameters
    ----------
    y_true : tf.Tensor
        A tensor of true labels with shape `(batch_size, max_string_length)`.
    y_pred : tf.Tensor
        A tensor of predictions with shape `(batch_size, time_steps,
        num_categories)`.

    Returns
    -------
    tf.Tensor
        The computed CTC loss for the batch.

    Notes
    -----
    This function calculates the CTC loss for each element in the batch by
    determining the length of each sequence and using the `ctc_batch_cost`
    function.
    """

    # Determine batch size and dimensions for input and label lengths
    batch_len = tf.shape(y_true, out_type=tf.int64)[0]
    input_length = tf.shape(y_pred, out_type=tf.int64)[1]
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

    # Create tensors for input length
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    # Calculate the CTC loss for each batch element
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)
