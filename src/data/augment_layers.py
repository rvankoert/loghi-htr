# Imports

# > Standard Library
import random
import logging

# > Third party libraries
import elasticdeform.tf as etf
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola
import tensorflow as tf
from data.gaussian_filter2d import gaussian_filter2d


class ShearXLayer(tf.keras.layers.Layer):
    def __init__(self, binary=False, **kwargs):
        super(ShearXLayer, self).__init__(**kwargs)
        self.fill_value = 1 if binary else 0

    @tf.function
    def call(self, inputs, training=None):
        """
        Apply shear transformation along the x-axis to the input image.

        This layer applies a shear transformation to the input image, creating
        an effect of slanting the shape of the image along the x-axis.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor. Can be either a 3-channel (RGB) or 4-channel
            (RGBA) image in tensor format.
        training : bool
            Whether layer will be used for training

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

        if not training:
            return inputs

        # Calculate random shear_factor
        shear_factor = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

        # Define the shear matrix
        shear_matrix = [1.0, shear_factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # Get the dynamic shape of the input tensor
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]

        # Ensure output shape is the same as input shape for height and width
        output_shape = [height, width]

        # Flatten the shear matrix for batch processing
        shear_matrix_tf = tf.reshape(
            tensor=tf.convert_to_tensor(shear_matrix, dtype=tf.float32),
            shape=[1, 8]
        )

        # Apply the shear transformation
        sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=inputs,
            transforms=shear_matrix_tf,
            output_shape=output_shape,
            fill_value=self.fill_value,
            fill_mode="CONSTANT",
            interpolation="NEAREST")

        return sheared_image


class ElasticTransformLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ElasticTransformLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=None):
        """
        Apply elastic transformation to an image tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            The original image tensor to be transformed. Can handle tensors of
            various shapes and channel numbers (e.g., grayscale, RGB, RGBA).
        training : bool
            Whether layer will be used for training

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

        if not training:
            return inputs

        # Hard set dtype to float32 because of compatibility
        inputs = tf.cast(inputs, tf.float32)

        # Random grid for elastic operation
        displacement_val = tf.random.normal([2, 3, 3]) * 5

        # Interpolation-order heavily influences result, operate on axis 1 and
        # 2, since 0 is batch and 3 is channel
        x_deformed = etf.deform_grid(
            inputs, displacement_val, axis=(1, 2), order=3,
            mode='nearest'
        )

        # Ensure output normalization for further augments
        x_deformed = tf.clip_by_value(x_deformed,
                                      clip_value_min=0.0,
                                      clip_value_max=1.0)

        # Ensure the output shape matches the input shape

        return x_deformed


class DistortImageLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DistortImageLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, training=None):
        """
        This layer applies random JPEG quality changes to each image in the
        batch. It supports different processing for RGB and RGBA images.

        Parameters
        ----------
        inputs : tf.Tensor
            Batch of input images as a TensorFlow tensor. The images should be
            in the format of either RGB or RGBA.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Batch of distorted images as a TensorFlow tensor, with the same
            shape and type as the input.
        """

        if not training:
            return inputs

        # Set data type to float32 for compatibility other layers
        inputs = tf.cast(inputs, tf.float32)

        def single_image_distort(img):
            # Process RGBA images
            if inputs.shape[-1] == 4:
                # Split RGB and Alpha channels
                rgb, alpha = img[..., :3], img[..., 3:]

                # Apply random JPEG quality to the RGB part
                rgb = tf.image.random_jpeg_quality(rgb,
                                                   min_jpeg_quality=50,
                                                   max_jpeg_quality=100)

                # Concatenate the RGB and Alpha channels back
                img = tf.concat([rgb, alpha], axis=-1)
            else:
                img = tf.image.random_jpeg_quality(img,
                                                   min_jpeg_quality=50,
                                                   max_jpeg_quality=100)
            return img

        # Apply the processing function to each image in the batch
        distorted_images = tf.map_fn(single_image_distort,
                                     inputs,
                                     fn_output_signature=inputs.dtype)
        return distorted_images


class RandomVerticalCropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomVerticalCropLayer, self).__init__(**kwargs)
        self.crop_factor = 0.75

    @tf.function
    def call(self, inputs, training=None):
        """
        Applies random crop to each image in the input batch.

        Parameters
        ----------
        inputs : tf.Tensor
            Input batch of images as a TensorFlow tensor.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Batch of images after applying random crop.
        """

        if not training:
            return inputs

        # Get the dynamic shape of the input tensor
        shape = tf.shape(inputs)
        height, width, channels = shape[1], shape[2], shape[3]

        # Generate a random crop factor
        random_crop = tf.random.uniform(shape=[], minval=0.6, maxval=1.0)
        self.crop_factor = random_crop

        crop_height = tf.cast(random_crop * tf.cast(height, tf.float32), tf.int32)

        # Define the crop size
        crop_size = tf.stack([crop_height, width, channels])

        # Generate random seeds for each image in the batch
        batch_size = tf.shape(inputs)[0]
        seeds = tf.random.uniform(shape=[batch_size, 2], minval=0, maxval=2**31 - 1, dtype=tf.int32)

        # Crop each image in the batch
        def crop_image(args):
            image, seed = args
            cropped = tf.image.stateless_random_crop(image, size=crop_size, seed=seed)
            return cropped

        # Apply the crop function to each image in the batch
        cropped_images = tf.map_fn(crop_image, (inputs, seeds), fn_output_signature=inputs.dtype)

        return cropped_images


class ResizeWithPadLayer(tf.keras.layers.Layer):
    def __init__(self, target_height, target_width=None, additional_width=None, binary=False, **kwargs):
        super(ResizeWithPadLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
        self.additional_width = additional_width
        if self.target_width is None and self.additional_width is None:
            raise ValueError("Either target_width or additional_width must be specified")
        self.binary = binary

    def estimate_background_color(self, image):
        """
        Estimate the background color of an image by analyzing the mean color
        of its edges and additional middle sections. This approach aims to
        provide a more accurate background color estimation by considering
        various segments of the image.

        Parameters
        ----------
        image : tf.Tensor
            Input image tensor in the format [batch_size, height, width,
            channels].

        Returns
        -------
        tf.Tensor
            The estimated background color as a tensor.
        """
        # Define slices for the edges
        top_edge = image[:, 0:1, :, :]
        bottom_edge = image[:, -1:, :, :]
        left_edge = image[:, :, 0:1, :]
        right_edge = image[:, :, -1:, :]

        # Calculate indices for the middle sections
        height, width = image.shape[1], image.shape[2]
        mid_height_start, mid_height_end = height // 4, 3 * height // 4
        mid_width_start, mid_width_end = width // 4, 3 * width // 4

        # Sample from middle sections of the edges
        left_mid_sample = image[:, mid_height_start:mid_height_end, 0:1, :]
        right_mid_sample = image[:, mid_height_start:mid_height_end, -1:, :]
        top_mid_sample = image[:, 0:1, mid_width_start:mid_width_end, :]
        bottom_mid_sample = image[:, -1:, mid_width_start:mid_width_end, :]

        # Calculate mean colors of the samples
        samples_mean = tf.concat([
            tf.reduce_mean(sample, axis=[1, 2, 3], keepdims=True)
            for sample in [top_edge, bottom_edge, left_edge, right_edge,
                           left_mid_sample, right_mid_sample,
                           top_mid_sample, bottom_mid_sample]
        ], axis=0)

        # Estimate background color by averaging all mean colors
        background_color = tf.reduce_mean(samples_mean, axis=0, keepdims=False)

        return background_color

    @tf.function
    def call(self, inputs, training=None):
        """
        Resize and pad the input images.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Resized and padded image tensor.
        """

        if not training:
            return inputs

        # Get the dynamic shape of the input
        shape = tf.shape(inputs)
        original_height, original_width = shape[1], shape[2]

        # Calculate target width
        if self.target_width is None:
            target_width = original_width + self.additional_width
        else:
            target_width = self.target_width

        # Resize image to the target height while maintaining aspect ratio
        resized_img = tf.image.resize(inputs, [self.target_height, target_width], preserve_aspect_ratio=True)

        # Get the shape of the resized image
        resized_shape = tf.shape(resized_img)
        resized_height, resized_width = resized_shape[1], resized_shape[2]

        # Calculate padding
        pad_height = self.target_height - resized_height
        pad_width = target_width - resized_width

        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        padding = [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]]

        # Estimate background color
        if self.binary:
            background_color = self.estimate_background_color(inputs)
            background_color_scalar = tf.reduce_mean(background_color)
        else:
            background_color_scalar = 0.0

        # Pad the image
        padded_img = tf.pad(resized_img, paddings=padding, mode="CONSTANT", constant_values=background_color_scalar)

        return padded_img


class BinarizeLayer(tf.keras.layers.Layer):
    def __init__(self, method='otsu', window_size=51, channels=None, **kwargs):
        super(BinarizeLayer, self).__init__(**kwargs)
        self.method = method

        if method == 'sauvola':
            logging.warning("Sauvola binarization is extremely inefficient, "
                            "select Otsu for faster processing")
        elif method != 'otsu':
            raise ValueError(
                "Invalid binarization method. Supported methods are 'otsu' "
                "and 'sauvola'")

        self.window_size = window_size
        self.channels = channels

    @tf.function
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
        input_shape = inputs.shape

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
    def __init__(self, channels=None, p=0.5, **kwargs):
        super(InvertImageLayer, self).__init__(**kwargs)
        self.channels = channels
        self.p = p

    @tf.function
    def call(self, inputs, training=None):
        """
        Inverts the pixel values of the input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Inverted image tensor.
        """

        if not training:
            return inputs

        # Randomly invert the image
        if random.random() > self.p:
            return inputs

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

    @tf.function
    def call(self, inputs, training=None):
        """
        Apply random Gaussian blur to the input tensor

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Blurred image tensor.
        """

        if not training:
            return inputs

        if self.mild_blur:
            blur_factor = 1
        else:
            blur_factor = round(random.uniform(0.1, 2), 1)
        return gaussian_filter2d(inputs,
                                 filter_shape=(11, 11),
                                 sigma=blur_factor)


class RandomWidthLayer(tf.keras.layers.Layer):
    def __init__(self, binary=False, **kwargs):
        super(RandomWidthLayer, self).__init__(**kwargs)
        self.fill_value = 1 if binary else 0

    @tf.function
    def call(self, inputs, training=None):
        """
        Randomly adjust the width of the input image.

        Parameters
        ----------
        inputs : tf.Tensor
            Input image tensor.
        training : bool
            Whether layer will be used for training

        Returns
        -------
        tf.Tensor
            Image tensor with adjusted width.
        """

        if not training:
            return inputs

        # Get the dynamic shape of the input
        shape = tf.shape(inputs)
        original_height = shape[1]
        original_width = shape[2]

        # Generate a random width scaling factor between 0.75 and 1.25
        random_width = tf.random.uniform(shape=[], minval=0.75, maxval=1.25)

        # Calculate the new width
        new_width = tf.cast(tf.cast(original_width, tf.float32) * random_width, tf.int32)

        # Resize image to new width while maintaining the original height
        resized_images = tf.image.resize(inputs, size=[original_height, new_width])

        # Pad or crop to original width
        diff = original_width - new_width
        pad_left = diff // 2
        pad_right = diff - pad_left

        padded_images = tf.cond(
            tf.greater(diff, 0),
            lambda: tf.pad(resized_images, [[0, 0], [0, 0], [pad_left, pad_right], [0, 0]], constant_values=self.fill_value),
            lambda: tf.image.crop_to_bounding_box(resized_images, 0, 0, original_height, original_width)
        )

        return padded_images
