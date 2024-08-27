# Imports

# > Standard library
import logging
import os
from typing import Any, List, Dict, Optional
import json
import shutil
import warnings
import zipfile

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from model.conversion import convert_model
from model.replacing import replace_final_layer, replace_recurrent_layer
from model.vgsl_model_generator import VGSLModelGenerator
from setup.config import Config

from model.custom_model import build_custom_model


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

    # Freeze layers if specified
    if any([config["freeze_conv_layers"],
            config["freeze_recurrent_layers"],
            config["freeze_dense_layers"]]):
        for layer in model.layers:
            if config["freeze_conv_layers"] and \
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
                              output_directory: Optional[str] = None,
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
    output_directory : Optional[str], optional
        The directory where the model should be saved after conversion, by
        default None.
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
        return convert_model(directory, output_directory, custom_objects)

    # Look for a .keras file
    model_file = next((os.path.join(directory, file) for file in os.listdir(
        directory) if file.endswith(".keras")), None)

    if model_file:
        try:
            return tf.keras.models.load_model(model_file,
                                              custom_objects=custom_objects,
                                              compile=compile)
        except (TypeError, ValueError):
            logging.error("Error loading model. Attempting to convert the "
                          "model to the new format.")

            # Convert the old model to the new format
            model = _convert_old_model_to_new(model_file, custom_objects,
                                              compile=compile)

            # Save the converted model
            # Rename the old model file
            if not os.path.exists(model_file + ".old"):
                logging.info("Renaming old model file to %s",
                             model_file + ".old")
                old_model_file = model_file + ".old"
                os.rename(model_file, old_model_file)

            # Save the new model
            logging.info("Saving new model to %s", model_file)
            model.save(model_file)

            return model

    raise FileNotFoundError("No suitable model file found in the directory.")


def _convert_old_model_to_new(model_file: str,
                              custom_objects: dict,
                              compile: bool = True) -> tf.keras.Model:
    """
    Converts an old v2 Keras model to the new v3 format.

    Parameters
    ----------
    model_file : str
        The path to the .keras file containing the old model.
    custom_objects : dict
        Custom objects required for model loading.
    compile : bool, optional
        Whether to compile the model after loading, by default True.

    Returns
    -------
    tf.keras.Model
        The converted Keras model.
    """

    # Temporary directory to extract the .keras file contents
    temp_dir = "/tmp/keras_model_extraction"

    # Ensure the temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Extract .keras file
    with zipfile.ZipFile(model_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Load model architecture from config.json
    with open(os.path.join(temp_dir, 'config.json'), 'r') as json_file:
        model_config = json.load(json_file)

    model_config["module"] = "keras.src.models.functional"

    # Function to recursively correct "axis" in BatchNormalization layers
    def correct_axis(layer_config):
        if isinstance(layer_config, dict):
            if layer_config.get('class_name') == 'BatchNormalization' \
                    and isinstance(layer_config.get('config', {})
                                   .get('axis'), list):
                layer_config['config']['axis'] \
                    = layer_config['config']['axis'][0]
            elif layer_config.get('class_name') == 'Bidirectional':
                lstm_layer_config = layer_config['config']['layer']['config']
                lstm_layer_config['recurrent_initializer']['class_name'] \
                    = 'OrthogonalInitializer'
                lstm_layer_config.pop('time_major', None)
            elif 'layers' in layer_config:
                for sub_layer in layer_config['layers']:
                    correct_axis(sub_layer)

    correct_axis(model_config["config"])

    # Replace 'Policy' with 'DTypePolicy' in all layers' dtype configurations
    def replace_policy_with_dtypepolicy(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'Policy':
                obj['class_name'] = 'DTypePolicy'
            for key, value in obj.items():
                replace_policy_with_dtypepolicy(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_policy_with_dtypepolicy(item)

    replace_policy_with_dtypepolicy(model_config["config"]['layers'])

    if model_config.get("compile_config"):
        compile_optimizer = model_config["compile_config"]["optimizer"]
        compile_optimizer["module"] = "keras.optimizers"
        compile_optimizer["config"].pop("jit_compile", None)
        compile_optimizer["config"].pop("is_legacy_optimizer", None)

    if not compile:
        model_config.pop("compile_config", None)

    # Save the corrected config back to a temporary json file
    corrected_config_path = os.path.join(temp_dir, 'corrected_config.json')
    with open(corrected_config_path, 'w') as json_file:
        json.dump(model_config, json_file)

    # Load model from corrected config
    with open(corrected_config_path, 'r') as json_file:
        corrected_model_json = json_file.read()
    model = tf.keras.models.model_from_json(
        corrected_model_json, custom_objects=custom_objects)

    # Load weights into the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_weights(os.path.join(temp_dir, 'model.weights.h5'),
                           skip_mismatch=True)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    return model


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
                                          output_directory=config["output"],
                                          custom_objects=custom_objects)
        if config["model_name"]:
            model._name = config["model_name"]
    elif config["model"] == 'custom':
        model = build_custom_model()
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
            .output.shape[2] - 2 - int(removed_padding)
    else:
        expected_length = model.get_layer(index=-1) \
            .output.shape[2] - 1 - int(removed_padding)
    if len(charlist) != expected_length:
        raise ValueError(
            f"Charlist length ({len(charlist)}) does not match "
            f"model output length ({expected_length}). If the charlist "
            "is correct, try setting use_mask to True.")
