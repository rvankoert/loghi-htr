# Imports

# > Standard library
import logging

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from utils import load_model_from_directory
from model import replace_final_layer, replace_recurrent_layer
from vgsl_model_generator import VGSLModelGenerator


def adjust_model_for_float32(model):
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


def customize_model(model, args, charlist):
    # Replace certain layers if specified
    if args.replace_recurrent_layer:
        logging.info("Replacing recurrent layer with "
                     f"{args.replace_recurrent_layer}")
        model = replace_recurrent_layer(model,
                                        len(charlist),
                                        args.replace_recurrent_layer,
                                        use_mask=args.use_mask)

    # Replace the final layer if specified
    if args.replace_final_layer:
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


def load_or_create_model(args, custom_objects, char_list):
    if args.existing_model:
        model = load_model_from_directory(
            args.existing_model, custom_objects=custom_objects)
        if args.model_name:
            model._name = args.model_name
    else:
        model_generator = VGSLModelGenerator(
            model=args.model,
            name=args.model_name,
            output_classes=len(char_list) + 2
            if args.use_mask else len(char_list) + 1
        )
        model = model_generator.build()

    return model


def verify_charlist_length(charlist, model, use_mask):
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


def summarize_model(model):
    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))
    return model_layers


def get_prediction_model(model):
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
