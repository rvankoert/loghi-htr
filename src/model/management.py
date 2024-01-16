# Imports

# > Standard library
import argparse
import logging
from typing import Any, List, Dict

import keras
import matplotlib.pyplot as plt
# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from utils.utils import load_model_from_directory
from data.augment_layers import *
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

    # Include data augments as separate Sequential object
    if any([args.aug_elastic_transform, args.aug_random_crop,
            args.aug_random_width, args.aug_distort_jpeg,
            args.aug_random_shear, args.aug_binarize_otsu,
            args.aug_binarize_sauvola, args.aug_blur, args.aug_invert,
            args.aug_random_augments]):
        # Set input params from trainings model input spec
        batch_size, width, height, channels = (model.layers[0]
                                               .get_input_at(0)
                                               .get_shape().as_list())
        # Retrieve the (random) augmentation model from the selected augments
        augment_options = get_augment_classes()
        augment_selection = get_augment_model()
        aug_model = make_augment_model(augment_options,
                                       augment_selection)

        if args.visualize_augments:
            # Plot augments on three different test images:
            for img_num in range(1, 4):
                # Save example plot locally with the pre and post from aug_model
                save_augment_steps_plot(
                    aug_model,
                    sample_image_path="../tests/data/test-image"
                                      + str(img_num)
                                      + ".png",
                    save_path="augment_test_img_"
                              + str(img_num)
                              + ".png",
                    channels=channels)
            logging.info("Augment visualizations are stored in the src folder")

        model = tf.keras.Sequential(aug_model.layers + model.layers)
        model.build(input_shape=[batch_size, height, width, channels])

    return model


def blend_with_background(image, background_color=None):
    """
    Blend the image with a background color. Assumes the image is in the
    format RGBA. This function is needed to correctly plot 4-channel images
    after binarization
    """
    if background_color is None:
        background_color = [1, 1, 1]
    rgb = image[..., :3]
    alpha = tf.expand_dims(image[..., 3], axis=-1)
    return rgb * alpha + background_color * (alpha - 1)


def save_augment_steps_plot(aug_model, sample_image_path, save_path, channels):
    """
    Applies each layer of an augmentation model to a sample image,
    plotting and saving the transformations sequentially.

    Parameters
    ----------
    aug_model : tf.keras.Sequential
        The augmentation model containing various layers.
    sample_image_path : str
        File path to the sample image.
    save_path : str
        Path where the plot of transformation steps is saved.
    channels : int
        Number of channels in the sample image.

    This function loads the sample image, applies each augmentation layer in
    the model sequentially, and plots the results, showing the effects of each
    step. The final plot is saved to the specified path.
    """

    # Load the sample image
    sample_image = tf.io.read_file(sample_image_path)
    sample_image = tf.image.decode_png(sample_image, channels=channels)
    sample_image = tf.image.convert_image_dtype(sample_image, dtype=tf.float32)
    sample_image = tf.expand_dims(sample_image, 0)  # Add batch dimension

    # Container for each step's image
    augment_images = [sample_image[0]]

    # Apply each augmentation layer to the image
    for layer in aug_model.layers:
        sample_image = layer(sample_image)
        augment_images.append(sample_image[0])

    # Plot the original and each augmentation step
    num_of_images = len(augment_images)
    plt.figure(figsize=(8, 1 * num_of_images))
    plt.suptitle("Data Augment steps:", fontsize=16)
    plt.axis('off')

    for idx, image in enumerate(augment_images):
        layer_name = aug_model.layers[idx - 1].name if idx > 0 else 'Original'
        image_shape = image.shape
        logging.debug("plotting layer_name: ", layer_name)
        logging.debug("plot image shape: ", image.shape)
        logging.debug("plot image dtype: ", image.dtype)

        # Adjust the image based on the number of channels
        if image.shape[-1] == 4:  # RGBA
            image = blend_with_background(image)
            cmap = None
        elif image.shape[-1] == 1:  # Grayscale
            image = tf.squeeze(image)
            cmap = 'gray'
        else:
            cmap = None

        # Ensure the image is of type float32 for plotting
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        plt.subplot(num_of_images, 1, idx + 1)
        plt.title(f'Step {idx}: {layer_name} {image_shape}')
        plt.tight_layout()
        plt.imshow(image, vmin=0, vmax=1, cmap=cmap)
        plt.axis('off')

    plt.savefig(save_path)


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
        expected_length = model.get_layer(('activation_1')) \
                              .get_output_at(0).shape[2] - 2
    else:
        expected_length = model.get_layer(('activation_1')) \
                              .get_output_at(0).shape[2] - 1

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
