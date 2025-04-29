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
    plotting and saving the transformations sequentially along with a pixel
    value histogram for debugging purposes.

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
    sample_image = tf.image.resize(sample_image, (64, 99999),
                                   preserve_aspect_ratio=True) / 255.0
    sample_image = tf.expand_dims(sample_image, 0)  # Add batch dimension

    # Container for each step's image
    augment_images = [sample_image[0]]

    # Calculate histogram for the original image
    histograms = []
    original_hist = tf.histogram_fixed_width(
        sample_image[0], [0.0, 1.0], nbins=256)
    histograms.append(original_hist)

    # Apply each augmentation layer to the image
    for layer in aug_model.layers:
        sample_image = layer(sample_image, training=True)
        augment_images.append(sample_image[0])

        # Ensure the tensor is converted to float32 for histogram calculation
        sample_image_float32 = tf.image.convert_image_dtype(sample_image[0],
                                                            dtype=tf.float32)

        # Calculate histogram for the augmented image
        hist = tf.histogram_fixed_width(
            sample_image_float32, [0.0, 1.0], nbins=256)
        histograms.append(hist)

    # Plot the original and each augmentation step
    num_of_images = len(augment_images)
    plt.figure(figsize=(20, 2 * num_of_images))
    plt.suptitle("Data Augment steps and histograms:", fontsize=16)
    plt.axis('off')

    for idx, (image, histogram) in enumerate(zip(augment_images, histograms)):
        layer_name = aug_model.layers[idx - 1].name if idx > 0 else 'Original'

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

        # Plotting the image
        ax = plt.subplot(num_of_images, 2, idx * 2 + 1)
        plt.title(f'Step {idx}: {layer_name}')
        plt.imshow(image, vmin=0, vmax=1, cmap=cmap)
        plt.axis('off')

        # Plotting the histogram
        ax = plt.subplot(num_of_images, 2, idx * 2 + 2)
        plt.title(f'Histogram {idx}: {layer_name}')
        plt.plot(histogram)
        plt.tight_layout()

    plt.savefig(save_path)


def visualize_augments(aug_model: tf.keras.Sequential,
                       output: str = "output",
                       channels: int = 3):
    """
    Visualize the effects of each augmentation step on a sample image.

    Parameters
    ----------
    aug_model : tf.keras.Sequential
        The augmentation model containing various layers.
    output : str
        Path to the output directory.
    channels : int
        Number of channels in the sample image.
    """

    root_dir = Path(__file__).resolve().parents[2]
    os.makedirs(output + "/augmentation-visualizations",
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
            save_path=output
            + "/augmentation-visualizations/test-img"
            + str(img_num)
            + ".png",
            channels=channels)
    logging.info("Augment visualizations are stored in the "
                 f"{output}/augmentation-visualizations/ "
                 "folder")


def get_augment_selection(config: Config, channels: int) -> list:
    """
    Construct a list of data augmentation layers based on the specified
    command-line arguments. Certain data augmentations like random_shear
    require some additional processing layers to ensure that information is
    not lost from the original image.

    Parameters
    ----------
    config: Config
        The Config object containing augmentation parameters
    channels: int
        Number of channels in the sample image

    Returns
    -------
    list
        A list of custom data augment layers that inherit from
        tf.keras.layers.Layer
    """

    augment_selection = []
    binarize_present = True if config["aug_binarize_sauvola"] or \
        config["aug_binarize_otsu"] else False

    if config["aug_distort_jpeg"]:
        logging.info("Selected data augment: JPEG distortion")
        augment_selection.append(DistortImageLayer())

    if config["aug_elastic_transform"]:
        logging.info("Selected data augment: elastic transform")
        augment_selection.append(ElasticTransformLayer())

    if config["aug_random_crop"]:
        logging.info("Selected data augment: random vertical crop")
        augment_selection.append(RandomVerticalCropLayer())

    if config["aug_random_width"]:
        logging.info("Selected data augment: random width")
        augment_selection.append(RandomWidthLayer(binary=binarize_present))

    # For some reason, the original adds a 50px pad to the width here
    augment_selection.append(ResizeWithPadLayer(target_height=64,
                                                additional_width=50,
                                                binary=binarize_present,
                                                name="extra_resize_with_pad"))

    if config["aug_random_shear"]:
        # Apply padding to make sure that shear does not cut off img
        augment_selection.append(ResizeWithPadLayer(target_height=64,
                                                    additional_width=64,
                                                    binary=binarize_present))

        logging.info("Selected data augment: random shear along x-axis")
        augment_selection.append(ShearXLayer(binary=binarize_present))

    if config["aug_binarize_sauvola"]:
        logging.info("Selected data augment: Sauvola binarization")
        augment_selection.append(BinarizeLayer(method='sauvola',
                                               channels=channels,
                                               window_size=51))

    if config["aug_binarize_otsu"]:
        logging.info("Selected data augment: Otsu binarization")
        augment_selection.append(BinarizeLayer(method="otsu",
                                               channels=channels))

    if config["aug_blur"]:
        logging.info("Selected data augment: image blur")
        augment_selection.append(BlurImageLayer())

    if config["aug_invert"]:
        logging.info("Selected data augment: invert image")
        augment_selection.append(InvertImageLayer(channels=channels))

    return augment_selection


def make_augment_model(config: Config, channels: int) -> tf.keras.Sequential:
    """
    Constructs an image augmentation model from a list of augmentation options,
    with specific handling for certain layer combinations. If no
    augment_selection is specified then a list of random augmentations are
    generated.

    Parameters
    ----------
    config: Config
        The Config object containing augmentation parameters.
    channels: int
        Number of channels in the sample image.

    Returns
    -------
    tf.keras.Sequential
        A TensorFlow Keras Sequential model comprising the selected
        augmentation layers.

    """

    selected_augmentations = get_augment_selection(config, channels)

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

    return tf.keras.Sequential(new_augment_selection,
                               name="data_augment_model")
