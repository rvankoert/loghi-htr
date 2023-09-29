# Imports

# > Standard Library

# > Local dependencies

# > Third party libraries
import tensorflow as tf
import keras
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    def __init__(self, filters, x, y, initializer=tf.initializers.GlorotNormal, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = layers.Conv2D(filters,
                                   kernel_size=(x, y),
                                   strides=(1, 1)
                                   if not downsample else (2, 2),
                                   padding="same",
                                   activation='elu',
                                   kernel_initializer=initializer)

        self.conv2 = layers.Conv2D(filters,
                                   kernel_size=(x, y),
                                   strides=(1, 1),
                                   padding="same",
                                   activation='elu',
                                   kernel_initializer=initializer)

        if downsample:
            self.conv3 = layers.Conv2D(filters,
                                       kernel_size=(1, 1),
                                       strides=(2, 2),
                                       padding="same",
                                       activation="elu",
                                       kernel_initializer=initializer)

        # ELU and BatchNormalization layers
        self.elu_layer = layers.ELU()
        self.bn_layer = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.conv1.filters,
            'x': self.conv1.kernel_size[0],
            'y': self.conv1.kernel_size[1],

            # Serializing the initializer
            'initializer': tf.keras.initializers.serialize(
                self.conv1.kernel_initializer),
            'downsample': self.downsample
        })
        return config

    def call(self, x, training=False):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.downsample:
            x = self.conv3(x)
        y = layers.Add()([x, y])
        y = self.elu_bn(y)
        return y

    def elu_bn(self, inputs):
        elu = self.elu_layer(inputs)
        bn = self.bn_layer(elu)
        return bn


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred
