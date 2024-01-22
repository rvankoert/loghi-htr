# Imports

# > Standard Library
import random
import logging

# Local dependencies
from setup.config import Config

# > Third party libraries
import elasticdeform.tf as etf
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola
import tensorflow as tf
import tensorflow_models as tfm


class ShearXLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply shear transformation along the x-axis to the input image.

        This layer applies a shear transformation to the input image, creating
        an effect of slanting the shape of the image along the x-axis.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor. Can be either a 3-channel (RGB) or 4-channel
            (RGBA) image in tensor format.

        Returns
        -------
        tf.Tensor
            The image tensor after applying shear transformation along the
            x-axis.

        Example
        -------
        >>> input_image = tf.random.uniform(shape=[1, 256, 256, 3])
        >>> shear_layer = ShearXLayer()
        >>> sheared_image = shear_layer(input_image)
        """

        # Calculate random shear_factor
        shear_factor = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

        # Define the shear matrix
        shear_matrix = [1.0, shear_factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # Get the dynamic shape of the input tensor
        input_shape = tf.shape(inputs)

        # Ensure output shape is the same as input shape for height and width
        output_shape = [input_shape[1], input_shape[2]]

        # Flatten the shear matrix for batch processing
        shear_matrix_tf = tf.reshape(
            tensor=tf.convert_to_tensor(shear_matrix, dtype=tf.dtypes.float32),
            shape=[1, 8]
        )

        # Apply the shear transformation
        # Fill value is set to 0 to ensure that binarization is not affected
        sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=inputs,
            transforms=shear_matrix_tf,
            output_shape=output_shape,
            fill_value=1,
            fill_mode="CONSTANT",
            interpolation="NEAREST")

        return sheared_image


class ElasticTransformLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply elastic transformation to an image tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            The original image tensor to be transformed. Can handle tensors of
            various shapes and channel numbers (e.g., grayscale, RGB, RGBA).

        Returns
        -------
        tf.Tensor
            The image tensor after applying elastic transformation.

        Notes
        -----
        - The `etf.deform_grid` function is used to apply the deformation. It
        allows control over the axis of deformation and the interpolation
        order.
        - The deformation values are clipped to ensure they stay within the
        [0, 1] range, which is important for subsequent image processing
        layers.

        Example
        -------
        >>> elastic_layer = ElasticTransformLayer()
        >>> original_image_tensor = tf.random.uniform(shape=[1, 256, 256, 3])
        >>> transformed_image_tensor = elastic_layer(original_image_tensor)
        """

        # Hard set dtype to float32 because of compatibility
        inputs = tf.cast(inputs, tf.float32)

        # Random grid for elastic operation
        displacement_val = tf.random.normal([2, 3, 3]) * 5

        # Interpolation-order heavily influences result, operate on axis 1 and
        # 2. cval is set to 1 to fill in with white so subsequent binarization
        # is still possible
        x_deformed = etf.deform_grid(
            inputs, displacement_val, axis=(1, 2), order=1,
            cval=1
        )

        # Ensure output normalization for further augments
        x_deformed = tf.clip_by_value(x_deformed,
                                      clip_value_min=0.0,
                                      clip_value_max=1.0)

        return x_deformed


class DistortImageLayer(tf.keras.layers.Layer):
    def __init__(self, channels=None, **kwargs):
        super(DistortImageLayer, self).__init__(**kwargs)
        self.channels = channels

    def call(self, inputs):
        """
        This layer applies random JPEG quality changes to each image in the
        batch. It supports different processing for RGB and RGBA images.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of input images as a TensorFlow tensor. The images should be
            in the format of either RGB or RGBA.

        Returns
        -------
        tf.Tensor
            Batch of distorted images as a TensorFlow tensor, with the same
            shape and type as the input.
        """

        # Set data type to float32 for compatibility other layers
        inputs = tf.cast(inputs, tf.float32)

        def single_image_distort(img):
            logging.debug("IMG SHAPE: ", img.shape)
            # Process RGBA images
            if self.channels == 4 or inputs.shape[-1] == 4:
                # Split RGB and Alpha channels
                rgb, alpha = img[..., :3], img[..., 3:]
                logging.debug("RGB SHAPE: ", rgb.shape)

                # Apply random JPEG quality to the RGB part
                rgb = tf.image.random_jpeg_quality(rgb,
                                                   min_jpeg_quality=50,
                                                   max_jpeg_quality=100)

                # Concatenate the RGB and Alpha channels back
                img = tf.concat([rgb, alpha], axis=-1)
            else:
                img = tf.image.random_jpeg_quality(img,
                                                   min_jpeg_quality=20,
                                                   max_jpeg_quality=100)
            return img

        # Apply the processing function to each image in the batch
        distorted_images = tf.map_fn(single_image_distort,
                                     inputs,
                                     dtype=inputs.dtype)
        return distorted_images


class RandomVerticalCropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomVerticalCropLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Applies random crop to each image in the input batch.

        Parameters
        ----------
        inputs : tf.Tensor
            Input batch of images as a TensorFlow tensor.

        Returns
        -------
        tf.Tensor
            Batch of images after applying random crop.
        """

        # Get the dtype of the input tensor
        input_dtype = inputs.dtype

        # Get the original height and width of the images
        input_shape = tf.shape(inputs)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Generate a random crop factor for each image in the batch
        min_crop_factor = tf.constant(0.6, dtype=tf.float32)
        crop_height = tf.cast(min_crop_factor * tf.cast(height,
                                                        tf.float32), tf.int32)

        # Define the crop size
        crop_size = (crop_height, width, channels)

        # Crop each image in the batch
        def crop_image(image):
            cropped = tf.image.random_crop(image, crop_size)

            # Cast back to the original dtype
            return tf.cast(cropped, input_dtype)

        # Ensure the output dtype matches the input
        cropped_images = tf.map_fn(crop_image, inputs, dtype=input_dtype)

        return cropped_images


class ResizeWithPadLayer(tf.keras.layers.Layer):
    def __init__(self, additional_width=50, **kwargs):
        super(ResizeWithPadLayer, self).__init__(**kwargs)
        self.additional_width = additional_width

    def call(self, inputs):
        """
        Resize and pad the input images.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.

        Returns
        -------
        tf.Tensor
            Resized and padded image tensor.
        """

        # Calculate the padding sizes
        padding_width = self.additional_width
        pad_left = padding_width // 2
        pad_right = padding_width - pad_left

        # Apply "white" padding
        padded_img = tf.pad(inputs,
                            [[0, 0], [0, 0], [pad_left, pad_right], [0, 0]],
                            mode='CONSTANT',
                            constant_values=1)

        return padded_img


class BinarizeLayer(tf.keras.layers.Layer):
    def __init__(self, method='otsu', window_size=51, channels=None, **kwargs):
        super(BinarizeLayer, self).__init__(**kwargs)
        self.method = method

        if method == 'sauvola':
            logging.warning("Sauvola is extremely inefficient, select "
                            "Otsu for faster processing")
        elif method != 'otsu':
            raise ValueError(
                "Invalid method. Supported methods are 'otsu' and 'sauvola'")

        self.window_size = window_size
        self.channels = channels

    def call(self, inputs):
        """
        Binarize the input tensor using either Otsu's or Sauvola's
        thresholding.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.

        Returns
        -------
        tf.Tensor
            Binarized image tensor.
        """

        # Check input shape
        input_shape = tf.shape(inputs)

        if input_shape[-1] == 1:
            # Just take grayscale if it is already 1-channel
            processed_input = inputs
        else:
            # Set image to grayscale (take first 3 channels)
            processed_input = tf.image.rgb_to_grayscale(inputs[..., :3])

        # Convert tensor to float32 for thresholding
        processed_input = tf.cast(processed_input, tf.float32)

        # Apply the selected thresholding method
        if self.method == 'otsu':
            def otsu_thresh(image):
                thresh = threshold_otsu(image)
                return ((image > thresh) * 1).astype(np.float32)

            binarized = tf.numpy_function(otsu_thresh, [processed_input],
                                          tf.float32)

        elif self.method == 'sauvola':
            def sauvola_thresh(image):
                thresh = threshold_sauvola(image, window_size=self.window_size)
                return ((image > thresh) * 1).astype(np.float32)

            binarized = tf.numpy_function(sauvola_thresh, [processed_input],
                                          tf.float32)

        # Ensure output shape is set correctly after numpy_function
        binarized.set_shape(processed_input.shape)

        # Repeat the grayscale image across 3 or 4 channels for compatibility
        if self.channels == 3:
            binarized = tf.tile(binarized, [1, 1, 1, 3])
        elif self.channels == 4:
            binarized = tf.tile(binarized, [1, 1, 1, 4])

        # Convert the binary image to float32
        return tf.cast(binarized, tf.float32)


class InvertImageLayer(tf.keras.layers.Layer):
    def __init__(self, channels=None, **kwargs):
        super(InvertImageLayer, self).__init__(**kwargs)
        self.channels = channels

    def call(self, inputs):
        """
        Inverts the pixel values of the input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.

        Returns
        -------
        tf.Tensor
            Inverted image tensor.
        """

        # Determine max value based on the data type of the input tensor
        if inputs.dtype.is_integer:
            max_value = tf.convert_to_tensor(255, dtype=inputs.dtype)
        else:
            max_value = tf.convert_to_tensor(1.0, dtype=inputs.dtype)

        # If the image has 4 channels, handle alpha channel separately
        if self.channels == 4:
            channels = tf.split(inputs, num_or_size_splits=4, axis=-1)
            inverted_channels = [max_value - channel for channel in
                                 channels[:3]]
            inverted_channels.append(channels[3])
            return tf.concat(inverted_channels, axis=-1)
        else:
            return max_value - inputs


class BlurImageLayer(tf.keras.layers.Layer):
    def __init__(self, mild_blur=False, **kwargs):
        super(BlurImageLayer, self).__init__(**kwargs)
        self.mild_blur = mild_blur

    def call(self, inputs):
        """
        Apply random Gaussian blur to the input tensor

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.

        Returns
        -------
        tf.Tensor
            Blurred image tensor.
        """

        if self.mild_blur:
            blur_factor = 1
        else:
            blur_factor = round(random.uniform(0.1, 2), 1)
        return tfm.vision.augment.gaussian_filter2d(inputs,
                                                    filter_shape=(10, 10),
                                                    sigma=blur_factor)


class RandomWidthLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Adjusts image width randomly and maintains original dimensions by
        either compressing or padding the image.

        Parameters
        ----------
        inputs : tf.Tensor
            A 3D or 4D tensor representing a single image or a batch of images.

        Returns
        -------
        tf.Tensor
            Processed image tensor with original height and width. The output
            might be altered due to resizing and padding.

        Notes
        -----
        - This method randomly scales the image's width between 75% and 125% of
          the original, then compresses or pads it to the original width.
        - Padding uses a constant value of 1 for compatibility with
          binarization
        """

        # Get the width and height of the input image
        if len(tf.shape(inputs)) < 4:
            # When input does have a batch size dim
            original_width = tf.shape(inputs)[1]
            original_height = tf.shape(inputs)[0]
        else:
            # When input does not have a batch size dim
            original_width = tf.shape(inputs)[2]
            original_height = tf.shape(inputs)[1]

        # Generate a random width scaling factor between 0.75 and 1.25
        random_width = tf.random.uniform(
            shape=[1], minval=0.75, maxval=1.25)[0]

        # Scale the width of the image by the random factor
        random_width *= float(original_width)
        new_width = int(random_width)

        # Convert the image to float32 dtype
        image = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

        # Resize image to the new width while maintaining the original height
        resized_image = tf.image.resize(image,
                                        size=[original_height, new_width])

        # Handling if new width is greater than original
        if new_width > original_width:
            # Resize back to the old width which causes minor compression
            compressed_image = tf.image.resize(resized_image,
                                               size=[original_height,
                                                     original_width])
            return compressed_image
        else:
            # Calculate padding size needed to match original width
            padding_width = original_width - new_width
            left_pad = padding_width // 2
            right_pad = padding_width - left_pad

            # Pad the new image to retain overall width for compatibility
            padded_image = tf.pad(resized_image,
                                  paddings=[[0, 0], [0, 0],
                                            [left_pad, right_pad],
                                            [0, 0]],
                                  mode="CONSTANT",
                                  constant_values=1)
            return padded_image


def get_augment_model(config: Config):
    """
    Construct a list of data augmentation layers based on the specified
    command-line arguments. Certain data augmentations like random_shear
    require some additional processing layers to ensure that information is
    not lost from the origian image.

    Parameters
    ----------
    config : Config
        The Config object containing augmentation parameters

    Returns
    -------
    list
        A list of custom data augment layers that inherit from
        tf.keras.layers.Layer
    """

    augment_selection = []

    if config["distort_jpegi"]:
        logging.info("Data augment: distort_jpeg")
        augment_selection.append(DistortImageLayer(channels=config["channels"]))

    if config["elastic_transform"]:
        logging.info("Data augment: elastic_transform")
        augment_selection.append(ElasticTransformLayer())

    if config["random_crop"]:
        logging.info("Data augment: random_crop")
        augment_selection.append(RandomVerticalCropLayer())

    if config["random_width"]:
        logging.info("Data augment: random_width")
        augment_selection.append(RandomWidthLayer())

    if config["do_binarize_sauvola"]:
        logging.info("Data augment: binarize_sauvola")
        augment_selection.append(BinarizeLayer(method='sauvola',
                                               channels=config["channels"],
                                               window_size=51))
    if config["do_binarize_otsu"]:
        logging.info("Data augment: binarize_otsu")
        augment_selection.append(BinarizeLayer(method="otsu",
                                               channels=config["channels"]))

    if config["do_random_shear"]:
        # Apply padding to make sure that shear does not cut off img
        augment_selection.append(ResizeWithPadLayer())

        logging.info("Data augment: shear_x")
        augment_selection.append(ShearXLayer())
        # Remove earlier padding to ensure correct output shapes
        augment_selection.append(tf.keras.layers.Cropping2D(cropping=(0, 25)))

    if config["do_blur"]:
        logging.info("Data augment: blur_image")
        augment_selection.append(BlurImageLayer())

    if config["do_invert"]:
        logging.info("Data augment: aug_invert")
        augment_selection.append(InvertImageLayer(channels=config["channels"]))

    return augment_selection


def make_augment_model(augment_selection=None):
    """
    Constructs an image augmentation model from a list of augmentation options,
    with specific handling for certain layer combinations. If no
    augment_selection is specified then a list of random augmentations are
    generated.

    Parameters
    ----------
    augment_selection : list, optional
        Preselected list of augmentation layers. If empty or None, a random
        selection is made.

    Returns
    -------
    tf.keras.Sequential
        A TensorFlow Keras Sequential model comprising the selected
        augmentation layers.

    """

    otsu_present = False
    new_augment_selection = []

    for aug_layer in augment_selection:
        if isinstance(aug_layer, BinarizeLayer):
            if aug_layer.method == 'otsu':
                otsu_present = True
            elif aug_layer.method == 'sauvola' and otsu_present:
                # Skip the 'sauvola' method if 'otsu' is present
                continue

        new_augment_selection.append(aug_layer)

    augment_selection = new_augment_selection

    # Init blur params
    mild_blur = False
    blur_index = -1

    # Check if a blur occurs before binarization, if so replace with mild blur
    for i, augment in enumerate(augment_selection):
        if isinstance(augment, BlurImageLayer):
            mild_blur = True
            blur_index = i
        elif mild_blur and isinstance(augment, BinarizeLayer):
            augment_selection[blur_index] = BlurImageLayer(mild_blur=True)

    # Check if blur/distort occurs after binarization, if so move it in front
    binarize = False
    binarize_index = -1
    for i, augment in enumerate(augment_selection[:]):
        if isinstance(augment, BinarizeLayer):
            binarize = True
            binarize_index = i
        elif binarize and (isinstance(augment, BlurImageLayer)):
            # Remove BlurImageLayer that occurs after binarization
            augment_selection.remove(augment)
            # Insert a mild blur object right before binarize
            augment_selection.insert(binarize_index,
                                     BlurImageLayer(mild_blur=True))
        elif binarize and (isinstance(augment, DistortImageLayer)):
            # Remove DistortImageLayer that occurs after binarization
            augment_selection.remove(augment)
            # Insert the Distort layer before the binarization
            augment_selection.insert(binarize_index, augment)

    adjusted_aug_model = tf.keras.Sequential(augment_selection,
                                             name="data_augment_model")
    return adjusted_aug_model
