# Imports

# > Standard library
import logging
import os
from pathlib import Path

# > Local dependencies
from data.augment_layers import BinarizeLayer, BlurImageLayer, \
    DistortImageLayer, ElasticTransformLayer, InvertImageLayer, \
    RandomVerticalCropLayer, RandomWidthLayer, ResizeWithPadLayer, ShearXLayer
from setup.config import Config

# > Third-party dependencies
import matplotlib.pyplot as plt
import tensorflow as tf


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


def save_augment_steps_plot(aug_model: tf.keras.Sequential,
                            sample_image_path: str,
                            save_path: str,
                            channels: int = 3) -> None:
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

    Notes
    -----
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
        sample_image = layer(sample_image, training=True)
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


def visualize_augments(aug_model: tf.keras.Sequential,
                       config: Config):
    """
    Visualize the effects of each augmentation step on a sample image.

    Parameters
    ----------
    aug_model : tf.keras.Sequential
        The augmentation model containing various layers.
    config : Config
        The Config object containing augmentation parameters.
    """

    root_dir = Path(__file__).resolve().parents[2]
    os.makedirs(config["output"] + "/augmentation-visualizations",
                exist_ok=True)

    logging.info("Visualizing augmentation steps...")

    # Plot augments on three different test images:
    for img_num in range(1, 4):
        # Save example plot locally with the pre and post from
        # aug_model
        save_augment_steps_plot(
            aug_model,
            sample_image_path=os.path.join(
                root_dir,
                "tests/data/test-image"
                + str(img_num)
                + ".png"),
            save_path=config["output"]
            + "/augmentation-visualizations/test-img"
            + str(img_num)
            + ".png",
            channels=config["channels"])
    logging.info("Augment visualizations are stored in the "
                 f"{config['output']}/augmentation-visualizations/ "
                 "folder")


def get_augment_selection(config: Config) -> list:
    """
    Construct a list of data augmentation layers based on the specified
    command-line arguments. Certain data augmentations like random_shear
    require some additional processing layers to ensure that information is
    not lost from the original image.

    Parameters
    ----------
    config: Config
        The Config object containing augmentation parameters

    Returns
    -------
    list
        A list of custom data augment layers that inherit from
        tf.keras.layers.Layer
    """

    augment_selection = []
    binarize_present = True if config["do_binarize_sauvola"] or \
        config["do_binarize_otsu"] else False

    if config["distort_jpeg"]:
        logging.info("Selected data augment: distort_jpeg")
        augment_selection.append(
            DistortImageLayer(channels=config["channels"]))

    if config["elastic_transform"]:
        logging.info("Selected data augment: elastic_transform")
        augment_selection.append(ElasticTransformLayer())

    if config["random_crop"]:
        logging.info("Selected data augment: random_crop")
        augment_selection.append(RandomVerticalCropLayer())

    if config["random_width"]:
        logging.info("Selected data augment: random_width")
        augment_selection.append(RandomWidthLayer(binary=binarize_present))

    if config["do_binarize_sauvola"]:
        logging.info("Selected data augment: binarize_sauvola")
        augment_selection.append(BinarizeLayer(method='sauvola',
                                               channels=config["channels"],
                                               window_size=51))

    if config["do_binarize_otsu"]:
        logging.info("Selected data augment: binarize_otsu")
        augment_selection.append(BinarizeLayer(method="otsu",
                                               channels=config["channels"]))

    if config["do_random_shear"]:
        # Apply padding to make sure that shear does not cut off img
        augment_selection.append(ResizeWithPadLayer(binary=binarize_present))

        logging.info("Selected data augment: shear_x")
        augment_selection.append(ShearXLayer(binary=binarize_present))
        # Remove earlier padding to ensure correct output shapes
        augment_selection.append(tf.keras.layers.Cropping2D(cropping=(0, 25)))

    if config["do_blur"]:
        logging.info("Selected data augment: blur_image")
        augment_selection.append(BlurImageLayer())

    if config["do_invert"]:
        logging.info("Selected data augment: aug_invert")
        augment_selection.append(InvertImageLayer(channels=config["channels"]))

    return augment_selection


def make_augment_model(config: Config) -> tf.keras.Sequential:
    """
    Constructs an image augmentation model from a list of augmentation options,
    with specific handling for certain layer combinations. If no
    augment_selection is specified then a list of random augmentations are
    generated.

    Parameters
    ----------
    config: Config
        The Config object containing augmentation parameters.

    Returns
    -------
    tf.keras.Sequential
        A TensorFlow Keras Sequential model comprising the selected
        augmentation layers.

    """

    selected_augmentations = get_augment_selection(config)

    otsu_present = False
    new_augment_selection = []

    for aug_layer in selected_augmentations:
        if isinstance(aug_layer, BinarizeLayer):
            if aug_layer.method == 'otsu':
                otsu_present = True
            elif aug_layer.method == 'sauvola' and otsu_present:
                # Skip the 'sauvola' method if 'otsu' is present
                continue

        new_augment_selection.append(aug_layer)

    selected_augmentations = new_augment_selection

    # Init blur params
    mild_blur = False
    blur_index = -1

    # Check if a blur occurs before binarization, if so replace with mild blur
    for i, augment in enumerate(selected_augmentations):
        if isinstance(augment, BlurImageLayer):
            mild_blur = True
            blur_index = i
        elif mild_blur and isinstance(augment, BinarizeLayer):
            selected_augmentations[blur_index] = BlurImageLayer(mild_blur=True)

    # Check if blur/distort occurs after binarization, if so move it in front
    binarize = False
    binarize_index = -1
    for i, augment in enumerate(selected_augmentations[:]):
        if isinstance(augment, BinarizeLayer):
            binarize = True
            binarize_index = i
        elif binarize and (isinstance(augment, BlurImageLayer)):
            # Remove BlurImageLayer that occurs after binarization
            selected_augmentations.remove(augment)
            # Insert a mild blur object right before binarize
            selected_augmentations.insert(binarize_index,
                                          BlurImageLayer(mild_blur=True))
        elif binarize and (isinstance(augment, DistortImageLayer)):
            # Remove DistortImageLayer that occurs after binarization
            selected_augmentations.remove(augment)
            # Insert the Distort layer before the binarization
            selected_augmentations.insert(binarize_index, augment)

    adjusted_aug_model = tf.keras.Sequential(selected_augmentations,
                                             name="data_augment_model")
    return adjusted_aug_model
