# Imports

# > Standard Library

# > Local dependencies

# > Third party libraries
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    """
    A residual block layer for a neural network.

    This layer applies two convolutional layers, each followed by batch
    normalization and an activation function.
    If downsampling is specified, it also applies a convolutional layer to
    the shortcut connection.

    Parameters
    ----------
    filters : int
        The number of filters for the convolutional layers.
    kernel_size : int or tuple of int
        The size of the kernel to use in the convolutional layers.
    initializer : tf.keras.initializers.Initializer,
                  default=tf.initializers.GlorotNormal()
        Initializer for the kernel weights of the convolutional layers.
    downsample : bool, default=False
        Whether to downsample the input feature maps using strides.
    activation : str, default='relu'
        Activation function to use after the convolutional layers.
    regularizer : tf.keras.regularizers.Regularizer, optional
        Regularizer function applied to the kernel weights of the
        convolutional layers.

    Attributes
    ----------
    conv1 : tf.keras.layers.Conv2D
        The first convolutional layer in the block.
    conv2 : tf.keras.layers.Conv2D
        The second convolutional layer in the block.
    bn1 : tf.keras.layers.BatchNormalization
        Batch normalization layer following conv1.
    bn2 : tf.keras.layers.BatchNormalization
        Batch normalization layer following conv2.
    downsample_conv : tf.keras.layers.Conv2D
        Convolutional layer used in the shortcut path for downsampling, if
        applicable.
    activation_layer : tf.keras.layers.Activation
        The activation layer applied after each batch normalization.

    Methods
    -------
    build(input_shape)
        Builds the residual block based on the input shape.
    call(inputs, training=False)
        Performs the forward pass on the inputs.
    get_config()
        Returns the configuration of the residual block.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 initializer=tf.initializers.GlorotNormal(),
                 downsample=False,
                 activation='relu',
                 regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.initializer = initializer
        self.downsample = downsample
        self.activation = activation
        self.regularizer = regularizer
        self.strides = 2 if downsample else 1

    def build(self, input_shape):
        """
        Create the layer based on input shape. This method creates the
        convolutional, batch normalization, and activation layers based on the
        input shape.

        Parameters
        ----------
        input_shape : TensorShape
            Shape of the input tensor.
        """

        self.conv1 = tf.keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=1,
            padding="same",
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if self.downsample:
            self.downsample_conv = tf.keras.layers.Conv2D(
                self.filters,
                (1, 1),
                strides=2,
                padding="same",
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer)

        self.activation_layer = tf.keras.layers.Activation(self.activation)

        super().build(input_shape)

    def call(self, inputs, training=False):
        """
        Perform the forward pass of the residual block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        training : bool, default=False
            Whether the layer should behave in training mode or inference mode.

        Returns
        -------
        tf.Tensor
            Output tensor of the residual block.
        """

        x = inputs
        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.activation_layer(y)

        y = self.conv2(y)
        y = self.bn2(y, training=training)

        if self.downsample:
            x = self.downsample_conv(x)

        y = tf.keras.layers.Add()([x, y])
        return self.activation_layer(y)

    def get_config(self):
        """
        Returns the configuration of the residual block for serialization.

        Returns
        -------
        dict
            A Python dictionary containing the configuration of the residual
            block.
        """

        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'downsample': self.downsample,
            'activation': self.activation,
            'regularizer': tf.keras.regularizers.serialize(self.regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a residual block from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the residual block.

        Returns
        -------
        ResidualBlock
            The residual block layer.
        """
        config['initializer'] = tf.keras.initializers.deserialize(
            config['initializer'])
        config['regularizer'] = tf.keras.regularizers.deserialize(
            config['regularizer'])

        return cls(**config)


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

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
