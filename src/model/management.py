# Imports

# > Standard library
import argparse
import logging
from typing import Any, List, Dict

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from utils.utils import load_model_from_directory
from model.model import replace_final_layer, replace_recurrent_layer
from model.vgsl_model_generator import VGSLModelGenerator


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
            logging.error(f"Layer {layer.name} is not float32")

    return model


def customize_model(model: tf.keras.Model, args: argparse.Namespace,
                    charlist: List[str]) -> tf.keras.Model:
    """
    Customizes a Keras model based on various arguments including layer
    replacement and freezing options.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be customized.
    args : argparse.Namespace
        A set of arguments controlling how the model should be customized.
    charlist : List[str]
        A list of characters used for model customization.

    Returns
    -------
    tf.keras.Model
        The customized model.
    """

    # Replace certain layers if specified
    if args.replace_recurrent_layer:
        logging.info("Replacing recurrent layer with "
                     f"{args.replace_recurrent_layer}")
        model = replace_recurrent_layer(model,
                                        len(charlist),
                                        args.replace_recurrent_layer,
                                        use_mask=args.use_mask)

    # Replace the final layer if specified
    if args.replace_final_layer or not args.existing_model:
        new_classes = len(charlist) + 2 if args.use_mask else len(charlist) + 1
        logging.info(f"Replacing final layer with {new_classes} classes")
        model = replace_final_layer(model, len(
            charlist), model.name, use_mask=args.use_mask)

    # Freeze or thaw layers if specified
    if any([args.thaw, args.freeze_conv_layers,
            args.freeze_recurrent_layers, args.freeze_dense_layers]):
        for layer in model.layers:
            if args.thaw:
                layer.trainable = True
                logging.info(f"Thawing layer: {layer.name}")
            elif args.freeze_conv_layers and \
                (layer.name.lower().startswith("conv") or
                 layer.name.lower().startswith("residual")):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False
            elif args.freeze_recurrent_layers and \
                    layer.name.lower().startswith("bidirectional"):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False
            elif args.freeze_dense_layers and \
                    layer.name.lower().startswith("dense"):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False

    # Further configuration based on use_float32 and gpu
    if args.use_float32 or args.gpu == "-1":
        # Adjust the model for float32
        logging.info("Adjusting model for float32")
        model = adjust_model_for_float32(model)

    return model


def load_or_create_model(args: argparse.Namespace,
                         custom_objects: Dict[str, Any]) -> tf.keras.Model:
    """
    Loads an existing Keras model or creates a new one based on provided
    arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to determine whether to load or create a model.
    custom_objects : Dict[str, Any]
        Custom objects required for model loading.

    Returns
    -------
    tf.keras.Model
        The loaded or newly created Keras model.
    """

    if args.existing_model:
        model = load_model_from_directory(
            args.existing_model, custom_objects=custom_objects)
        if args.model_name:
            model._name = args.model_name
    else:
        model_generator = VGSLModelGenerator(
            model=args.model,
            name=args.model_name,
            channels=args.channels
        )
        model = model_generator.build()

    return model


def verify_charlist_length(charlist: List[str], model: tf.keras.Model,
                           use_mask: bool) -> None:
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

    Raises
    ------
    ValueError
        If the length of the charlist does not match the expected output length
        of the model.
    """

    # Verify that the length of the charlist is correct
    if use_mask:
        expected_length = model.layers[-1].output_shape[2] - 2
    else:
        expected_length = model.layers[-1].output_shape[2] - 1
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
