# Imports

# > Standard library
import logging
import os
from typing import Any, List, Dict, Optional

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from model.replacing import replace_final_layer, replace_recurrent_layer
from model.vgsl_model_generator import VGSLModelGenerator
from setup.config import Config


def adjust_model_for_float32(model: tf.keras.Model) -> tf.keras.Model:
    """
    Adjusts a given Keras model to use float32 data type for all layers.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be adjusted for float32.

    Returns
    -------
    tf.keras.Model
        A new model with all layers set to use float32 data type.
    """

    # Recreate the exact same model but with float32
    config = model.get_config()

    # Set the dtype policy for each layer in the configuration
    for layer_config in config['layers']:
        if 'dtype' in layer_config['config']:
            layer_config['config']['dtype'] = 'float32'
        if 'dtype_policy' in layer_config['config']:
            layer_config['config']['dtype_policy'] = {
                'class_name': 'Policy',
                'config': {'name': 'float32'}}

    # Create a new model from the modified configuration
    model_new = tf.keras.Model.from_config(config)
    model_new.set_weights(model.get_weights())
    model = model_new

    # Verify float32
    for layer in model.layers:
        if layer.dtype != 'float32':
            logging.error("Layer %s is not float32", layer.name)

    return model


def customize_model(model: tf.keras.Model,
                    config: Config,
                    charlist: List[str]) -> tf.keras.Model:
    """
    Customizes a Keras model based on various arguments including layer
    replacement and freezing options.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be customized.
    config : Config
        A set of arguments controlling how the model should be customized.
    charlist : List[str]
        A list of characters used for model customization.

    Returns
    -------
    tf.keras.Model
        The customized model.
    """

    # Replace certain layers if specified
    if config["replace_recurrent_layer"]:
        logging.info("Replacing recurrent layer with %s",
                     config["replace_recurrent_layer"])
        model = replace_recurrent_layer(model,
                                        len(charlist),
                                        config["replace_recurrent_layer"],
                                        use_mask=config["use_mask"])

    # Replace the final layer if specified
    if config["replace_final_layer"] or not os.path.isdir(config["model"]):
        new_classes = len(charlist) + \
            2 if config["use_mask"] else len(charlist) + 1
        logging.info("Replacing final layer with %s classes", new_classes)
        model = replace_final_layer(model, len(charlist), model.name,
                                    use_mask=config["use_mask"])

    # Freeze or thaw layers if specified
    if any([config["thaw"], config["freeze_conv_layers"],
            config["freeze_recurrent_layers"], config["freeze_dense_layers"]]):
        for layer in model.layers:
            if config["thaw"]:
                layer.trainable = True
                logging.info("Thawing layer: %s", layer.name)
            elif config["freeze_conv_layers"] and \
                (layer.name.lower().startswith("conv") or
                 layer.name.lower().startswith("residual")):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False
            elif config["freeze_recurrent_layers"] and \
                    layer.name.lower().startswith("bidirectional"):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False
            elif config["freeze_dense_layers"] and \
                    layer.name.lower().startswith("dense"):
                logging.info("Freezing layer: %s", layer.name)
                layer.trainable = False

    # Further configuration based on use_float32 and gpu
    if config["use_float32"] or config["gpu"] == "-1":
        # Adjust the model for float32
        logging.info("Adjusting model for float32")
        model = adjust_model_for_float32(model)

    return model


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
        return tf.keras.saving.load_model(directory,
                                          custom_objects=custom_objects,
                                          compile=compile)

    # Look for a .keras file
    model_file = next((os.path.join(directory, file) for file in os.listdir(
        directory) if file.endswith(".keras")), None)

    if model_file:
        return tf.keras.saving.load_model(model_file,
                                          custom_objects=custom_objects,
                                          compile=compile)

    raise FileNotFoundError("No suitable model file found in the directory.")


def load_or_create_model(config: Config,
                         custom_objects: Dict[str, Any]) -> tf.keras.Model:
    """
    Loads an existing Keras model or creates a new one based on provided
    arguments.

    Parameters
    ----------
    config : Config
        The configuration object containing the arguments.
    custom_objects : Dict[str, Any]
        Custom objects required for model loading.

    Returns
    -------
    tf.keras.Model
        The loaded or newly created Keras model.
    """

    # Check if config["model"] is a directory
    if os.path.isdir(config["model"]):
        model = load_model_from_directory(config["model"],
                                          custom_objects=custom_objects)
        if config["model_name"]:
            model._name = config["model_name"]
    else:
        model_generator = VGSLModelGenerator(
            model_spec=config["model"],
            name=config["model_name"],
            channels=config["channels"]
        )
        model = model_generator.build()

    return model


def verify_charlist_length(charlist: List[str],
                           model: tf.keras.Model,
                           use_mask: bool,
                           removed_padding: bool) -> None:
    """
    Verifies if the length of the character list matches the expected output
    length of the model.

    Parameters
    ----------
    charlist : List[str]
        List of characters to be verified.
    model : tf.keras.Model
        The model whose output length is to be checked.
    use_mask : bool
        Indicates whether a mask is being used or not.
    removed_padding : bool
        Indicates whether padding was removed from the character list.

    Raises
    ------
    ValueError
        If the length of the charlist does not match the expected output length
        of the model.
    """

    # Verify that the length of the charlist is correct
    if use_mask:
        expected_length = model.get_layer(index=-1) \
            .get_output_at(0).shape[2] - 2 - int(removed_padding)
    else:
        expected_length = model.get_layer(index=-1) \
            .get_output_at(0).shape[2] - 1 - int(removed_padding)
    if len(charlist) != expected_length:
        raise ValueError(
            f"Charlist length ({len(charlist)}) does not match "
            f"model output length ({expected_length}). If the charlist "
            "is correct, try setting use_mask to True.")


def get_prediction_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Extracts a prediction model from a given Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The complete Keras model from which the prediction model is to be
        extracted.

    Returns
    -------
    tf.keras.Model
        The prediction model extracted from the given model, typically up to
        the last dense layer.

    Raises
    ------
    ValueError
        If no dense layer is found in the given model.
    """

    last_dense_layer = None
    for layer in reversed(model.layers):
        if layer.name.startswith('dense'):
            last_dense_layer = layer
            break
    if last_dense_layer is None:
        raise ValueError("No dense layer found in the model")

    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, last_dense_layer.output
    )
    return prediction_model
