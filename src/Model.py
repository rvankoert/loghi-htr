import os

import tensorflow as tf
from keras.applications.xception import Xception
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Lambda, Dense
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.keras.layers import Add,Concatenate, Conv2D, ELU, GlobalMaxPooling2D, MaxPooling2D, ReLU, BatchNormalization, AveragePooling2D
from tensorflow import Tensor
from LoghiCustomCallback import LoghiCustomCallback

import keras.backend as K
import json

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

epsilon = backend_config.epsilon

def elu_bn(inputs: Tensor) -> Tensor:
    elu = ELU()(inputs)
    bn = BatchNormalization()(elu)
    return bn


def residual_block(x, downsample, filters, kernel_size, initializer) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides=((1, 1) if not downsample else (2, 2)),
               filters=filters,
               padding="same",
               activation='elu',
               kernel_initializer=initializer)(x)
    y = Conv2D(kernel_size=kernel_size,
               strides=(1, 1),
               filters=filters,
               padding="same",
               activation='elu',
               kernel_initializer=initializer)(y)
    # y = Conv2D(kernel_size=kernel_size,
    #            strides=(1, 1),
    #            filters=filters,
    #            padding="same",
    #            activation='elu',
    #            kernel_initializer=initializer)(y)

    if downsample:
        x = Conv2D(kernel_size=(1, 1),
                   strides=(2, 2),
                   filters=filters,
                   padding="same",
                   activation='elu',
                   kernel_initializer=initializer)(x)
    out = Add()([x, y])
    out = elu_bn(out)
    return out

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

    y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

    return array_ops.expand_dims(
        ctc.ctc_loss(
            inputs=y_pred,
            labels=sparse_labels,
            sequence_length=input_length,
            ignore_longer_outputs_than_inputs=True),
        1)


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    @tf.function
    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # print("CTC lambda inputs / shape")
        # print("y_pred:", y_pred.shape)  # (?, 778, 30)
        # print("labels:", y_true.shape)  # (?, 80)
        # print("input_length:", input_length.shape)  # (?, 1)
        # print("label_length:", label_length.shape)  # (?, 1)
        # print("loss:", loss)  # (?, 1)

        # At test time, just return the computed predictions
        return y_pred


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """

    def __init__(self, name='CER_metric', greedy=True, beam_width=1, **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")
        self.greedy = greedy
        self.beam_width = beam_width

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        decode, log = K.ctc_decode(y_pred,
                                   input_length,
                                   greedy=True,
                                   beam_width=10)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, 0))
        y_true_sparse = tf.sparse.retain(y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=False)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(K.cast(tf.size(y_true_sparse.values), 'float32'))

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
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                   input_length,
                                   greedy=True)

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        y_true_sparse = tf.sparse.retain(y_true_sparse, tf.not_equal(y_true_sparse.values, 0))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(K.cast(len(y_true), 'float32'))
        # self.counter.assign_add(10)

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_state(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)


@tf.function
def CTCLoss(y_true, y_pred):
    # # Compute the training-time loss value
    # y_true = tf.where(tf.equal(y_true, 01), tf.ones_like(y_true), y_true)
    # y_true = tf.replace(y_true, tf.not_equal(y_true, -1))
    # y_true = tf.sparse.retain(y_true, tf.not_equal(y_true, 0))
    # y_pred = tf.sparse.retain(y_pred, tf.not_equal(y_pred.values, 0))
    # y_pred = tf.where(tf.equal(y_pred, -1), tf.zeros_like(y_pred), y_pred)
    # total_length = tf.size(y_true.values)

    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    # input_length = tf.math.count_nonzero(y_pred, axis=-1, keepdims=True)
    # label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

    # https://stackoverflow.com/questions/64321779/how-to-use-tf-ctc-loss-with-variable-length-features-and-labels
    # tf.print(input_length)
    # tf.print(label_length)
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    # label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class Model:

    def __init__(self):
        pass

    @staticmethod
    def replace_recurrent_layer(model, number_characters, use_mask=False, use_gru=False,
                                rnn_layers=2, rnn_units=256, use_rnn_dropout=True, dropout_rnn=0.5):
        initializer = tf.keras.initializers.GlorotNormal()
        last_layer = ""
        for layer in model.layers:
            if layer.name.startswith('bidirectional_'):
                break
            last_layer = layer.name

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name=last_layer).output
        )
        if not use_rnn_dropout:
            dropout_rnn = 0
        x = prediction_model.output
        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                    dropout=dropout_rnn,
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}",
                                        dropout=dropout_rnn,
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            # if use_rnn_dropout:
            #     if i < rnn_layers:
            #         x = layers.Dropout(rate=dropout_rnn)(x)

        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=prediction_model.inputs, outputs=output, name="model_new8"
        )

        return model

    @staticmethod
    def replace_final_layer(model, number_characters, model_name, use_mask=False):
        initializer = tf.keras.initializers.GlorotNormal()
        last_layer = ""
        for layer in model.layers:
            if layer.name == "dense3":
                break
            last_layer = layer.name

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name=last_layer).output
        )
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(prediction_model.output)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(prediction_model.output)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=prediction_model.inputs, outputs=output, name=model_name
        )

        return model

    @staticmethod
    def set_dropout(model, dropout=0.5):
        initializer = tf.keras.initializers.GlorotNormal()
        last_layer = ""
        for layer in model.layers:
            if layer.name.startswith('dropout'):
                layer.rate = dropout

        return model


    @staticmethod
    def build_model_new17(img_size, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True,
                          dropout_dense=0.5,
                          dropout_conv=0.0, dropout_rnn=0.5, dropout_recurrent_dropout=0.5, seed=42):
        (height, width, channels) = img_size[0], img_size[1], img_size[2]

        padding = "same"
        activation = "elu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal(seed)
        channel_axis = -1

        num_filters = 16

        t = BatchNormalization()(input_img)
        t = Conv2D(kernel_size=[3, 3],
                   strides=(1, 1),
                   filters=num_filters,
                   padding="same",
                   activation='elu',
                   kernel_initializer=initializer)(t)

        num_blocks_list = [2, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters,
                                   kernel_size=[3, 3], initializer=initializer)
            num_filters *= 2

        x = t
        new_shape = (-1, x.shape[-2] * x.shape[-1])

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    dropout=dropout_rnn,
                    recurrent_dropout=dropout_recurrent_dropout,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",

                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}",
                                        dropout=dropout_rnn,
                                        recurrent_dropout=dropout_recurrent_dropout
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout_dense > 0:
            x = layers.Dropout(dropout_dense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new14"
        )
        return model

    @staticmethod
    def build_model_new16(img_size, number_characters,
                          use_mask=False,
                          use_gru=False,
                          rnn_layers=5,
                          rnn_units=128,
                          batch_normalization=False,
                          use_rnn_dropout=True,
                          dropout_dense=0.5,
                          dropout_conv=0.0,
                          dropout_rnn=0.5,
                          dropout_recurrent_dropout=0,
                          seed=42,
                          multiplier=4
                          ):
        (height, width, channels) = img_size[0], img_size[1], img_size[2]

        padding = "same"
        activation = "elu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        initializer = tf.keras.initializers.GlorotNormal(seed)
        channel_axis = -1

        x = input_img

        # First conv block
        x = layers.Conv2D(
            filters=2 * multiplier,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        # Second conv block
        x = layers.Conv2D(
            filters=3 * multiplier,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            5 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            8 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            12 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        # x = layers.Conv2D(
        #     196,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # if use_dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        oldshape = (-1, x.shape[-2], 512)
        print(oldshape)
        # oldshape = x.shape
        new_shape = (-1, x.shape[-2] * x.shape[-1])

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    dropout=dropout_rnn,
                    recurrent_dropout=dropout_recurrent_dropout,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",

                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}",
                                        dropout=dropout_rnn,
                                        recurrent_dropout=dropout_recurrent_dropout
                                        )

            x = recurrent(x)

        print(x.shape)
        x = layers.Reshape(target_shape=new_shape, name="reshapeOld")(x)
        # x = layers.Conv2D(
        #     filters=2 * multiplier,
        #     kernel_size=[3, 3],
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="ConvExtra",
        #     kernel_initializer=initializer
        # )(x)
        # x = layers.Reshape(target_shape=new_shape, name="reshapeNew")(x)
        #
        # for i in range(1, rnn_layers + 1):
        #     if use_gru:
        #         recurrent = layers.GRU(
        #             units=rnn_units,
        #             # activation=activation,
        #             recurrent_activation="sigmoid",
        #             dropout=dropout_rnn,
        #             recurrent_dropout=dropout_recurrent_dropout,
        #             unroll=False,
        #             use_bias=True,
        #             return_sequences=True,
        #             kernel_initializer=initializer,
        #             reset_after=True,
        #             name=f"gru_{i}",
        #
        #         )
        #     else:
        #         recurrent = layers.LSTM(rnn_units,
        #                                 # activation=activation,
        #                                 return_sequences=True,
        #                                 kernel_initializer=initializer,
        #                                 name=f"lstm2_{i}",
        #                                 dropout=dropout_rnn,
        #                                 recurrent_dropout=dropout_recurrent_dropout
        #                                 )
        #
        #     x = layers.Bidirectional(
        #         recurrent, name=f"bidirectional2_{i}", merge_mode="concat"
        #     )(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout_dense > 0:
            x = layers.Dropout(dropout_dense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new16"
        )
        return model

    @staticmethod
    def build_model_new15(img_size, number_characters,
                          use_mask=False,
                          use_gru=False,
                          rnn_layers=5,
                          rnn_units=128,
                          batch_normalization=False,
                          use_rnn_dropout=True,
                          dropout_dense=0.5,
                          dropout_conv=0.0,
                          dropout_rnn=0.5,
                          dropout_recurrent_dropout=0,
                          seed=42,
                          multiplier=4
                          ):
        (height, width, channels) = img_size[0], img_size[1], img_size[2]

        padding = "same"
        activation = "elu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        initializer = tf.keras.initializers.GlorotNormal(seed)
        channel_axis = -1

        x = input_img

        # First conv block
        x = layers.Conv2D(
            filters=2 * multiplier,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        # Second conv block
        x = layers.Conv2D(
            filters=3 * multiplier,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            5 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            8 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            12 * multiplier,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout_conv > 0:
            x = layers.Dropout(dropout_conv)(x)

        # x = layers.Conv2D(
        #     196,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # if use_dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    dropout=dropout_rnn,
                    recurrent_dropout=dropout_recurrent_dropout,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",

                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}",
                                        dropout=dropout_rnn,
                                        recurrent_dropout=dropout_recurrent_dropout
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout_dense > 0:
            x = layers.Dropout(dropout_dense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new15"
        )
        return model

    @staticmethod
    def build_model_new14(img_size, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True,
                          dropout_dense=0.5,
                          dropout_conv=0.0, dropout_rnn=0.5, dropout_recurrent_dropout=0.5, seed=42):
        (height, width, channels) = img_size[0], img_size[1], img_size[2]

        padding = "same"
        activation = "elu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal(seed)
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropout_conv)(x)

        # Second conv block
        x = layers.Conv2D(
            filters=36,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout:
            x = layers.Dropout(dropout_conv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        if dropout:
            x = layers.Dropout(dropout_conv)(x)

        # x = layers.Conv2D(
        #     196,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    dropout=dropout_rnn,
                    recurrent_dropout=dropout_recurrent_dropout,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",

                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}",
                                        dropout=dropout_rnn,
                                        recurrent_dropout=dropout_recurrent_dropout
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout_dense > 0:
            x = layers.Dropout(dropout_dense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new14"
        )
        return model

    @staticmethod
    def build_model_new13(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True, dropoutdense=0.5,
                          dropoutconv=0.0, dropout_rnn=0.5):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        # dropoutdense = 0
        # dropoutconv = 0
        # dropoutdense = 0.5
        # dropoutconv = 0.0
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=12,
            kernel_size=[1, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1a",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 1],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1b",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            filters=36,
            kernel_size=[1, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2a",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Conv2D(
            filters=48,
            kernel_size=[3, 1],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2b",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            64,
            (1, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3a",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Conv2D(
            96,
            (3, 1),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3b",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (1, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4a",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Conv2D(
            96,
            (3, 1),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4b",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=dropout_rnn)(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout:
            x = layers.Dropout(dropoutdense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new13"
        )
        return model

    @staticmethod
    def build_model_new12(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True, dropoutdense=0.5,
                          dropoutconv=0.0, dropout_rnn=0.5):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=12,
            kernel_size=[1, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            filters=48,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     8*96,
        #     (1, 8),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding='valid',
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # # x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=dropout_rnn)(x)

        # x = layers.Dense(1024, activation="elu",
        #                  kernel_initializer=initializer)(x)
        if dropout:
            x = layers.Dropout(dropoutdense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new12"
        )
        return model

    @staticmethod
    def build_model_new11(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True, dropoutdense=0.5,
                          dropoutconv=0.0, dropout_rnn=0.5):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=48,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     192,
        #     (1, 8),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding='valid',
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # # x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=dropout_rnn)(x)

        x = layers.Dense(1024, activation="elu",
                         kernel_initializer=initializer)(x)
        if dropout:
            x = layers.Dropout(dropoutdense)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new11"
        )
        return model

    @staticmethod
    def build_model_new10(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                          batch_normalization=False, dropout=False, use_rnn_dropout=True, seed=42):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal(seed=seed)
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=48,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            96,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new10"
        )
        return model

    @staticmethod
    def build_model_new9(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=False, dropout=False, use_rnn_dropout=True):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model

        dropoutconv = 0.0
        dropout_rnn = 0.5
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=12,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool5")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     256,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool6")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        # x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        # x = layers.ReLU(name="dense_1_relu")(x)
        # if dropout:
        #     x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new9"
        )
        return model

    @staticmethod
    def build_model_new8(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=False, dropout=False, use_rnn_dropout=True):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=12,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool5")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     256,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool6")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if use_rnn_dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        # x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        # x = layers.ReLU(name="dense_1_relu")(x)
        # if dropout:
        #     x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new8"
        )
        return model

    @staticmethod
    def build_model_new7(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=False, dropout=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        x = input_img

        # if use_mask:
        #     masked = x

        # First conv block
        x = layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool5")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     256,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation=activation,
        #     padding=padding,
        #     name="Conv6",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same', name="pool6")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # if use_mask:
        #     x = Lambda(lambda x: x, output_shape=lambda s: s)(x)

        # if use_mask:
        #
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # if use_mask:
        #     masked = layers.MaxPooling2D(pool_size=(3, 3), strides=(8, 8), padding='same', name="maskpool1")(masked)
        #     new_shape = (-1, masked.shape[-2] * masked.shape[-1])
        #     masked = layers.Reshape(target_shape=new_shape, name="reshape_mask")(masked)
        #     x = layers.Multiply()([x, masked])
        #     x = tf.keras.layers.Masking(mask_value=-10.0)(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        # x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        # x = layers.ReLU(name="dense_1_relu")(x)
        # if dropout:
        #     x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new7"
        )
        return model

    @staticmethod
    def build_model_new6(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=True):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropout_rnn = 0.0
        padding = "same"
        width = None

        base_model = Xception(include_top=False, weights='imagenet', input_shape=(width, height, channels))
        base_model.trainable = False
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        x = base_model(input_img, training=False)
        x = layers.Conv2D(
            256,
            (1, 1),
            strides=(1, 1),
            padding=padding,
            name="Conv4",
        )(x)

        # # dropout1 = Dropout(0.5)(flat1)
        # flat2 = Dense(1024, name="fc_dense1", activation="elu")(flat1)
        # dropout2 = Dropout(0.5)(flat2)
        # class1 = Dense(1024, name="fc_dense2", activation="elu")(dropout2)
        # dropout3 = Dropout(0.5)(class1)
        # output = Dense(numClasses, activation='softmax')(dropout3)
        # # define new model
        # for layer in model.layers[:126]:
        #     if layer.name == 'fc_dense1':
        #         break
        #     layer.trainable = False

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1
        dropout = False
        # x = layers.BatchNormalization(name="conv_2_bn")(input_img)

        # x = model.layers[-1].output

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        if dropout:
            x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new6"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_new5(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=True, dropout=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        activation = "relu"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1

        # x = layers.BatchNormalization(name="conv_2_bn")(input_img)
        x = input_img
        # First conv block
        x = layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation=activation,
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool4")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # x = layers.Dense(128, activation="elu", name="dense1")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutdense)(x)
        # x = layers.Dense(128, activation="elu", name="dense2")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutdense)(x)

        # x = layers.Conv2D(
        #     1536,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool5")(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     2048,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        if use_mask:
            x = tf.keras.layers.Masking()(x)
        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation=activation,
                    # recurrent_activation="sigmoid",
                    # recurrent_dropout=0,
                    # unroll=False,
                    # use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    # reset_after=True,
                    name=f"gru_{i}"
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        # activation=activation,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        # x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        # x = layers.ReLU(name="dense_1_relu")(x)
        # if dropout:
        #     x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new5"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_new4(imgSize, number_characters, use_mask=False, use_gru=False, rnn_layers=5, rnn_units=128,
                         batch_normalization=True):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.0
        dropout_rnn = 0.0
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        channel_axis = -1
        dropout = False
        # x = layers.BatchNormalization(name="conv_2_bn")(input_img)
        x = input_img
        # First conv block
        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(input_img)
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(2, 2),
            activation='elu',
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool1")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool2")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis)(x)
        # x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="pool3")(x)
        if dropout:
            x = layers.Dropout(dropoutconv)(x)

        # # Second conv block
        # x = layers.Conv2D(
        #     64,
        #     (3, 3),
        #     strides=(1, 1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv4",
        #     kernel_initializer=initializer
        # )(x)
        # if batch_normalization:
        #     x = layers.BatchNormalization(axis=channel_axis)(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv5",
        #     kernel_initializer=initializer
        # )(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # if dropout:
        #     x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, x.shape[-2] * x.shape[-1])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

        for i in range(1, rnn_layers + 1):
            if use_gru:
                recurrent = layers.GRU(
                    units=rnn_units,
                    # activation="tanh",
                    # recurrent_activation="sigmoid",
                    # recurrent_dropout=0,
                    # unroll=False,
                    # use_bias=True,
                    return_sequences=True,
                    dropout=dropout_rnn,
                    kernel_initializer=initializer,
                    # reset_after=True,
                    name=f"gru_{i}",
                )
            else:
                recurrent = layers.LSTM(rnn_units,
                                        return_sequences=True,
                                        dropout=dropout_rnn,
                                        kernel_initializer=initializer,
                                        name=f"lstm_{i}"
                                        )

            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if dropout:
                if i < rnn_layers:
                    x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        if dropout:
            x = layers.Dropout(rate=0.5)(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="model_new4"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    # xception based
    @staticmethod
    def build_model_xception3(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        channel_axis = -1
        # height = 128;
        # width = height;
        batch_normalization = False

        input_img = layers.Input(
            shape=(None, height, channels), name="image"
        )

        x = layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          use_bias=False,
                          name='block1_conv1')(input_img)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('elu', name='block1_conv1_act')(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('elu', name='block1_conv2_act')(x)

        residual = layers.Conv2D(128, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(728, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('elu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv1')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('elu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv2')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('elu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv3')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('elu', name='block14_sepconv2_act')(x)

        x = layers.Conv2D(
            256,
            (1, 1),
            strides=(1, 1),
            activation='elu',
            name="conv_final",
        )(x)

    @staticmethod
    def build_model_new3(imgSize, output_dim, rnn_layers=5, rnn_units=128):
        # based on https://keras.io/examples/audio/ctc_asr/
        """Model similar to DeepSpeech2."""
        # Model's input
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        input_img = layers.Input(
            shape=(height, None, channels), name="image"
        )
        # labels = layers.Input(name="label", shape=(None,))

        # input_spectrogram = layers.Input((None, input_dim), name="input")
        # Expand the dimension to use 2D CNN.
        # x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_img)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(input_img)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        # x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        x = layers.Reshape((-1, x.shape[-3] * 16))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        # x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        # x = layers.ReLU(name="dense_1_relu")(x)
        # x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax", name="dense3")(x)
        # Model
        model = keras.Model(input_img, output, name="DeepSpeech_2")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)
        return model

    @staticmethod
    def build_model_new2(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.0
        dropout_rnn = 0.0
        padding = "same"
        width = None
        channel_axis = -1
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(input_img)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 128)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # x = layers.Dropout(dropoutdense)(x)
        # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # x = layers.Dropout(dropoutdense)(x)

        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(x)

        if use_gru:
            x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
        else:
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        # x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = layers.Dense(units=number_characters + 1, activation="softmax")(x)
        output = x
        # Model
        model = keras.Model(input_img, output, name="DeepSpeech_2")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)
        return model

    @staticmethod
    def build_model_new1(imgSize, input_dim, output_dim, rnn_layers=5, rnn_units=128):
        # based on https://keras.io/examples/audio/ctc_asr/
        """Model similar to DeepSpeech2."""
        # Model's input
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        # labels = layers.Input(name="label", shape=(None,))

        # input_spectrogram = layers.Input((None, input_dim), name="input")
        # Expand the dimension to use 2D CNN.
        # x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_img)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(input_img)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        model = keras.Model(input_img, output, name="DeepSpeech_2")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)
        return model

    # xception based
    @staticmethod
    def build_model(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        channel_axis = -1
        # height = 128;
        # width = height;
        batch_normalization = False

        input_img = layers.Input(
            shape=(None, height, channels), name="image"
        )

        x = layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          use_bias=False,
                          name='block1_conv1')(input_img)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('elu', name='block1_conv1_act')(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('elu', name='block1_conv2_act')(x)

        residual = layers.Conv2D(128, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(728, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('elu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv1')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('elu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv2')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('elu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv3')(x)
            if batch_normalization:
                x = layers.BatchNormalization(axis=channel_axis,
                                              name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        if batch_normalization:
            residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('elu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv1')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation('elu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv2')(x)
        if batch_normalization:
            x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('elu', name='block14_sepconv2_act')(x)

        x = layers.Conv2D(
            256,
            (1, 1),
            strides=(1, 1),
            activation='elu',
            name="conv_final",
        )(x)

        # x = layers.GlobalMaxPooling2D()(x)

        # new_shape = (-1, width//299, 2048)
        # # new_shape = (-1, (height) * 128)
        # # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # # x = layers.Dropout(dropoutdense)(x)
        # # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # # x = layers.Dropout(dropoutdense)(x)

        # x = layers.Reshape((-1, 4096))(x) # 103
        # x = layers.Reshape((-1, 2048))(x)
        # x = layers.Reshape((-1, 6144))(x)
        x = layers.Reshape((-1, 256 * 3))(x)

        dropout_rnn = 0
        initializer = tf.keras.initializers.GlorotNormal()
        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                            kernel_initializer=initializer))(x)
        x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                            kernel_initializer=initializer))(x)
        # x = layers.Bidirectional(layers.GRU(512,
        #                                         activation="tanh",
        #                                         recurrent_activation="sigmoid",
        #                                         use_bias=True,
        #                                         return_sequences=True,
        #                                         dropout=dropout_rnn,
        #                                         reset_after=True))(x)
        # x = layers.Bidirectional(layers.GRU(512,
        #                                         activation="tanh",
        #                                         recurrent_activation="sigmoid",
        #                                         use_bias=True,
        #                                         return_sequences=True,
        #                                         dropout=dropout_rnn,
        #                                         reset_after=True,
        #                                         ))(x)

        # Output layer
        x = layers.Dense(number_characters + 2, activation="softmax", name="dense3", )(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    # xception based
    @staticmethod
    def build_model_old6(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        channel_axis = -1
        # height = 128;
        # width = height;

        input_img = layers.Input(
            shape=(None, height, channels), name="image"
        )

        x = layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          use_bias=False,
                          name='block1_conv1')(input_img)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)

        residual = layers.Conv2D(128, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv2D(128, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block2_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv2D(728, (1, 1),
                                 strides=(2, 2),
                                 padding='same',
                                 use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv1')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                          name=prefix + '_sepconv1_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv2')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                          name=prefix + '_sepconv2_bn')(x)
            x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = layers.SeparableConv2D(728, (3, 3),
                                       padding='same',
                                       use_bias=False,
                                       name=prefix + '_sepconv3')(x)
            x = layers.BatchNormalization(axis=channel_axis,
                                          name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                                 padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization(axis=channel_axis)(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_sepconv2_act')(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block13_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv2D(2048, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block14_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)

        # x = layers.GlobalMaxPooling2D()(x)

        # new_shape = (-1, width//299, 2048)
        # # new_shape = (-1, (height) * 128)
        # # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        # x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # # x = layers.Dropout(dropoutdense)(x)
        # # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # # x = layers.Dropout(dropoutdense)(x)

        x = layers.Reshape((-1, 2048))(x)

        dropout_rnn = 0.0
        initializer = tf.keras.initializers.GlorotNormal()
        # x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                            kernel_initializer=initializer))(x)
        x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                            kernel_initializer=initializer))(x)
        # x = layers.Bidirectional(layers.GRU(512,
        #                                         activation="tanh",
        #                                         recurrent_activation="sigmoid",
        #                                         use_bias=True,
        #                                         return_sequences=True,
        #                                         dropout=dropout_rnn,
        #                                         reset_after=True))(x)
        # x = layers.Bidirectional(layers.GRU(512,
        #                                         activation="tanh",
        #                                         recurrent_activation="sigmoid",
        #                                         use_bias=True,
        #                                         return_sequences=True,
        #                                         dropout=dropout_rnn,
        #                                         reset_after=True,
        #                                         ))(x)

        # Output layer
        x = layers.Dense(number_characters + 2, activation="softmax", name="dense3", )(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_old5(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutdense = 0.5
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )

        # labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()
        # x = layers.BatchNormalization(name="conv_2_bn")(input_img)

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 128)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(1024, activation="elu", name="dense1")(x)
        x = layers.Dropout(dropoutdense)(x)
        x = layers.Dense(1024, activation="elu", name="dense2")(x)
        x = layers.Dropout(dropoutdense)(x)

        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(x)

        if use_gru:
            x = layers.Bidirectional(layers.GRU(512,
                                                activation="tanh",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                return_sequences=True,
                                                reset_after=True,
                                                dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.GRU(512,
                                                activation="tanh",
                                                recurrent_activation="sigmoid",
                                                use_bias=True,
                                                return_sequences=True,
                                                reset_after=True,
                                                dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
        else:
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        output = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        # output = CTCLayer(name="ctc_loss")(x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_old4(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.1
        dropout_rnn = 0.5
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 128)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # x = layers.Dropout(dropoutdense)(x)
        # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # x = layers.Dropout(dropoutdense)(x)

        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(x)

        if use_gru:
            x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
        else:
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_old3(imgSize, number_characters, use_mask=False, use_gru=False):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.0
        dropout_rnn = 0.0
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        labels = layers.Input(name="label", shape=(None,))
        initializer = tf.keras.initializers.GlorotNormal()

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
            kernel_initializer=initializer
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
            kernel_initializer=initializer
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
            kernel_initializer=initializer
        )(x)
        x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 128)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # x = layers.Dropout(dropoutdense)(x)
        # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # x = layers.Dropout(dropoutdense)(x)

        if use_mask:
            x = tf.keras.layers.Masking(mask_value=-1.0)(x)

        if use_gru:
            x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.GRU(512, return_sequences=True, dropout=dropout_rnn,
                                                kernel_initializer=initializer))(x)
        else:
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)
            x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn,
                                                 kernel_initializer=initializer))(x)

        # Output layer
        if use_mask:
            x = layers.Dense(number_characters + 2, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        else:
            x = layers.Dense(number_characters + 1, activation="softmax", name="dense3",
                             kernel_initializer=initializer)(x)
        x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_old2(imgSize, number_characters, learning_rate):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutconv = 0.0
        dropout_rnn = 0.0
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        labels = layers.Input(name="label", shape=(None,))

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            128,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
        )(x)
        x = layers.Dropout(dropoutconv)(x)
        #
        # x = layers.Conv2D(
        #     128,
        #     (3, 3),
        #     strides=(1,1),
        #     activation='elu',
        #     padding=padding,
        #     name="Conv6",
        # )(x)
        # x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 128)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        # x = layers.Dense(1024, activation="elu", name="dense1")(x)
        # x = layers.Dropout(dropoutdense)(x)
        # x = layers.Dense(1024, activation="elu", name="dense2")(x)
        # x = layers.Dropout(dropoutdense)(x)

        # x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        # x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=dropout_rnn))(x)
        # x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=dropout_rnn))(x)
        # x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=dropout_rnn))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        # x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn))(x)
        x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=dropout_rnn))(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

        # Output layer
        x = layers.Dense(number_characters + 1, activation="softmax", name="dense3")(x)

        x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    @staticmethod
    def build_model_old1(imgSize, number_characters, learning_rate):
        (height, width, channels) = imgSize[0], imgSize[1], imgSize[2]
        # Inputs to the model
        dropoutdense = 0.5
        dropoutconv = 0.1
        padding = "same"
        width = None
        input_img = layers.Input(
            shape=(width, height, channels), name="image"
        )
        labels = layers.Input(name="label", shape=(None,))

        # First conv block
        x = layers.Conv2D(
            16,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1")(x)
        x = layers.Dropout(dropoutconv)(x)
        # Second conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            48,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv3",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3")(x)
        x = layers.Dropout(dropoutconv)(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv4",
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        x = layers.Conv2D(
            80,
            (3, 3),
            strides=(1, 1),
            activation='elu',
            padding=padding,
            name="Conv5",
        )(x)
        x = layers.Dropout(dropoutconv)(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        # new_shape = ((width // 4), (height // 4) * 64)

        new_shape = (-1, (height // 8) * 80)
        # new_shape = (-1, (height) * 128)
        # x = tf.reshape(input, shape=[73, (height // 4) * 64])
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(1024, activation="elu", name="dense1")(x)
        x = layers.Dropout(dropoutdense)(x)
        x = layers.Dense(1024, activation="elu", name="dense2")(x)
        x = layers.Dropout(dropoutdense)(x)

        # x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        # x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=dropout))(x)
        # x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        # x = tf.keras.layers.Masking(mask_value=-1.0)(x)
        # x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.LSTM(256, return_sequences=True, dropout=0.5)(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

        # Output layer
        x = layers.Dense(number_characters + 1, activation="softmax", name="dense3")(x)

        x = layers.Activation('linear', dtype=tf.float32)(x)
        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # # Optimizer
        # # opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1.0)
        # opt = keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.1)
        # # Compile the model and return
        # model.compile(optimizer=opt)
        return model

    # # Train the model
    @staticmethod
    def train_batch(model, train_dataset, validation_dataset, epochs, output, model_name, steps_per_epoch=None,
                    early_stopping_patience=20, num_workers=20, max_queue_size=256, output_checkpoints=False,
                    metadata=None, charlist=None):
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
        # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.3, cooldown=2, patience=5,
                                           verbose=1, min_delta=1e-4, mode='min')
        callbacks.append(history)
        # callbacks.append(mcp_save)
        # if output_checkpoints:
        #     filepath = output + '/checkpoints/' + model_name + "-saved-model-{epoch:02d}-{loss:.4f}"
        #     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')
        #     callbacks.append(checkpoint)

        callbacks.append(LoghiCustomCallback( save_best=True, save_checkpoint=output_checkpoints, output=output,
                                              charlist=charlist, metadata=metadata))
        filename = os.path.join(output, 'log.csv')
        history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
        callbacks.append(history_logger)

        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            # batch_size=1,
            callbacks=callbacks,
            shuffle=True,
            workers=num_workers,
            max_queue_size=max_queue_size,
            steps_per_epoch=steps_per_epoch
        )
        return history
