# Imports

# > Standard library
import logging
import os

# > Local dependencies
from model.custom_callback import LoghiCustomCallback
from model.vgsl_model_generator import VGSLModelGenerator

# > Third party dependencies
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops import math_ops, array_ops, ctc_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.keras import backend_config

logger = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

epsilon = backend_config.epsilon


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Runs CTC loss algorithm on each batch element.
  Arguments:
      y_true: tensor `(samples, max_string_length)`
          containing the truth labels.
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_pred`.
      label_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_true`.
  Returns:
      Tensor with shape (samples,1) containing the
          CTC loss of each element.
  """
    label_length = math_ops.cast(
        array_ops.squeeze(label_length, axis=-1), dtypes_module.int32)
    input_length = math_ops.cast(
        array_ops.squeeze(input_length, axis=-1), dtypes_module.int32)
    sparse_labels = math_ops.cast(
        K.ctc_label_dense_to_sparse(y_true, label_length), dtypes_module.int32)

    y_pred = math_ops.log(array_ops.transpose(
        y_pred, perm=[1, 0, 2]) + epsilon())

    return array_ops.expand_dims(
        ctc_ops.ctc_loss(
            inputs=y_pred,
            labels=sparse_labels,
            sequence_length=input_length,
            ignore_longer_outputs_than_inputs=True),
        1)


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """

    def __init__(self, name='CER_metric', greedy=True, beam_width=1, **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(
            name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")
        self.greedy = greedy
        self.beam_width = beam_width

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        decode, log = K.ctc_decode(y_pred,
                                   input_length,
                                   greedy=True,
                                   beam_width=10)

        decode = K.ctc_label_dense_to_sparse(
            decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(
            y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, 0))
        y_true_sparse = tf.sparse.retain(
            y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=False)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(
            K.cast(tf.size(y_true_sparse.values), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Error Rate
    """

    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(
            name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                   input_length,
                                   greedy=True)

        decode = K.ctc_label_dense_to_sparse(
            decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(
            y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        y_true_sparse = tf.sparse.retain(
            y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        correct_words_amount = tf.reduce_sum(
            tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(K.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_state(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def replace_recurrent_layer(model: keras.Model,
                            number_characters: int,
                            vgsl_string: str,
                            use_mask: bool = False) -> keras.Model:
    """
    Replace recurrent layers in a given Keras model with new layers specified
    by a VGSL string.

    Parameters
    ----------
    model : keras.Model
        The original Keras model in which recurrent layers need to be replaced.
    number_characters : int
        The number of characters/classes for the Dense layer before the
        activation.
    vgsl_string : str
        The VGSL spec string that defines the new layers to replace the
        recurrent ones.
    use_mask : bool, optional
        Whether to use masking for the Dense layer. If True, an additional unit
        is added.
        Default is False.

    Returns
    -------
    keras.Model
        A new Keras model with the recurrent layers replaced by those specified
        in the VGSL string.

    Raises
    ------
    ValueError:
        If no recurrent layers are found in the given model.
    """

    logging.info("Starting the replacement of recurrent layers in the model.")

    initializer = tf.keras.initializers.GlorotNormal()

    # Identify layers up to the first recurrent layer
    last_layer = None
    found_rnn = False
    for layer in model.layers:
        if isinstance(layer, (layers.GRU, layers.LSTM, layers.Bidirectional)):
            found_rnn = True
            break
        last_layer = layer

    if not found_rnn:
        logging.error("No recurrent layers found in the model.")
        raise ValueError("No recurrent layers found in the model.")

    # Generate new layers using VGSLModelGenerator
    logging.info("Generating new layers using VGSLModelGenerator.")
    vgsl_gen = VGSLModelGenerator(vgsl_string)

    logging.debug(f"VGSLModelGenerator history: {vgsl_gen.history}")

    # Add the new layers to the model
    x = last_layer.output
    for layer_name in vgsl_gen.history:
        new_layer = getattr(vgsl_gen, layer_name)
        x = new_layer(x)

    dense_layer_name = model.layers[-2].name
    if use_mask:
        x = layers.Dense(number_characters + 2,
                         activation="softmax",
                         name=dense_layer_name,
                         kernel_initializer=initializer)(x)
    else:
        x = layers.Dense(number_characters + 1,
                         activation="softmax",
                         name=dense_layer_name,
                         kernel_initializer=initializer)(x)
    output = layers.Activation('linear', dtype=tf.float32)(x)

    old_model_name = model.name
    model = keras.models.Model(
        inputs=model.input, outputs=output, name=old_model_name
    )

    logging.info(
        f"Recurrent layers replaced successfully in model: {old_model_name}.")

    return model


def replace_final_layer(model, number_characters, model_name, use_mask=False):
    initializer = tf.keras.initializers.GlorotNormal()
    last_layer = ""
    for layer in model.layers:
        if layer.name.startswith("dense") or \
                layer.name.startswith("activation"):
            continue
        last_layer = layer.name

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(
            name=last_layer).output
    )
    if use_mask:
        x = layers.Dense(number_characters + 2, activation="softmax", name="dense_out",
                         kernel_initializer=initializer)(prediction_model.output)
    else:
        x = layers.Dense(number_characters + 1, activation="softmax", name="dense_out",
                         kernel_initializer=initializer)(prediction_model.output)
    output = layers.Activation('linear', dtype=tf.float32)(x)
    model = keras.models.Model(
        inputs=prediction_model.inputs, outputs=output, name=model_name
    )

    return model


# # Train the model

def train_batch(model, train_dataset, validation_dataset, epochs, lr_schedule,
                output, model_name, steps_per_epoch=None,
                early_stopping_patience=20, num_workers=20, max_queue_size=256, output_checkpoints=False,
                metadata=None, charlist=None, verbosity_mode='auto'):
    # # Add early stopping
    callbacks = []
    if early_stopping_patience > 0 and validation_dataset:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_CER_metric',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)
    from keras.callbacks import History
    from keras.callbacks import ModelCheckpoint
    history = History()
    if validation_dataset:
        base_path = output + '/best_val/'
        mcp_save = ModelCheckpoint(base_path, save_best_only=True, monitor='val_CER_metric',
                                   mode='min', verbose=1)
    else:
        mcp_save = ModelCheckpoint(output + '/best_train/', save_best_only=True, monitor='CER_metric',
                                   mode='min', verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.3, cooldown=2, patience=5,
                                       verbose=1, min_delta=1e-4, mode='min')
    callbacks.append(history)
    callbacks.append(LoghiCustomCallback(save_best=True, save_checkpoint=output_checkpoints, output=output,
                                         charlist=charlist, metadata=metadata))
    filename = os.path.join(output, 'log.csv')
    history_logger = tf.keras.callbacks.CSVLogger(
        filename, separator=",", append=True)
    callbacks.append(history_logger)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        workers=num_workers,
        max_queue_size=max_queue_size,
        steps_per_epoch=steps_per_epoch,
        verbose=verbosity_mode
    )
    return history
