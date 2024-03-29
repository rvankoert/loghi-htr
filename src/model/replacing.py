# Imports

# > Standard library
import logging

# > Third party dependencies
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# > Local dependencies
from model.vgsl_model_generator import VGSLModelGenerator


def replace_recurrent_layer(model: tf.keras.Model,
                            number_characters: int,
                            vgsl_string: str,
                            use_mask: bool = False) -> tf.keras.Model:
    """
    Replace recurrent layers in a given Keras model with new layers specified
    by a VGSL string.

    Parameters
    ----------
    model : tf.keras.Model
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
    tf.keras.Model
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

    logging.debug("VGSLModelGenerator history: %s", vgsl_gen.history)

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

    logging.info("Recurrent layers replaced successfully in model: %s.",
                 old_model_name)

    return model


def replace_final_layer(model: tf.keras.models.Model,
                        number_characters: int,
                        model_name: str,
                        use_mask: bool = False) -> tf.keras.models.Model:
    """
    Replace the final layer of a given Keras model.

    This function replaces the final dense layer of a model with a new dense
    layer that has a specified number of units. An optional mask can be used
    which affects the number of units in the new dense layer.

    Parameters
    ----------
    model : tf.keras.models.Model
        The Keras model whose final layer is to be replaced.
    number_characters : int
        Number of units for the new dense layer.
    model_name : str
        Name to assign to the modified model.
    use_mask : bool, optional
        Whether to use a mask, which adds two additional units to the layer, by
        default False.

    Returns
    -------
    tf.keras.models.Model
        The modified Keras model with a new final layer.
    """

    # Initialize the kernel weights
    initializer = tf.keras.initializers.GlorotNormal()

    # Find the name of the last layer before the dense or activation layers
    last_layer = ""
    for layer in model.layers:
        if not layer.name.startswith(("dense", "activation")):
            last_layer = layer.name

    # Create a prediction model up to the last layer
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name=last_layer).output
    )

    # Add a new dense layer with adjusted number of units based on use_mask
    if use_mask:
        units = number_characters + 2
    else:
        units = number_characters + 1

    x = layers.Dense(units, activation="softmax", name="dense_out",
                     kernel_initializer=initializer)(prediction_model.output)

    # Add a linear activation layer with float32 data type
    output = layers.Activation('linear', dtype=tf.float32)(x)

    # Construct the final model
    new_model = tf.keras.models.Model(
        inputs=prediction_model.inputs, outputs=output, name=model_name
    )

    return new_model
