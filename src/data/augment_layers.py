# Imports

# > Standard Library
import random


# > Local dependencies


# > Third party libraries
import tensorflow as tf
import elasticdeform.tf as etf
import tensorflow_models as tfm
from skimage.filters import threshold_otsu, threshold_sauvola
import numpy as np
import matplotlib.pyplot as plt


class ShearXLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply shear transformation along the x-axis to the input image.

        Parameters
        ----------
        - image (numpy.ndarray): Input image, can be 3-channel (RGB) or
          4-channel (RGBA).

        Returns
        -------
        - numpy.ndarray: Sheared image.

        Example
        -------
        >>> input_image = np.random.rand(256, 256, 3)
        >>> sheared_result = shear_x(input_image, 0.5)
        """

        # Calculate random shear_factor
        shear_factor = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

        # Define the shear matrix
        shear_matrix = [1.0, shear_factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # Add batch dimension to the input image
        image_rgb = tf.expand_dims(inputs, axis=0)

        # Flatten the shear matrix for batch processing
        shear_matrix_tf = tf.reshape(
            tf.convert_to_tensor(shear_matrix,
                                 dtype=tf.dtypes.float32), [1, 8])

        # Fill value
        fill_value = tf.convert_to_tensor(0.0, dtype=tf.float32)

        # Apply the shear transformation on the GPU
        sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=image_rgb,
            transforms=shear_matrix_tf,
            output_shape=tf.constant([inputs.shape[0], inputs.shape[1]],
                                     dtype=tf.int32),
            fill_value=fill_value,
            interpolation="BILINEAR"
        )

        # Remove the batch dimension from the resulting tensor
        sheared_image = tf.squeeze(sheared_image, axis=0)
        return sheared_image


class ElasticTransformLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply elastic transformation to an image.

        Parameters
        ----------
        - original (tf.Tensor): The original image to be transformed.

        Returns
        -------
        - tf.Tensor: The image after elastic transformation.

        Notes
        -----
        - Elastic transformation introduces local deformations to the image to
          enhance robustness and variability in the dataset.
        - It uses a random displacement field generated from a normal
          distribution.
        - The `etf.deform_grid` function is employed for the deformation,
          allowing control over the axis and interpolation order.

        Example
        -------
        ```python
        transformer = ImageTransformer()
        original_image = load_image("path/to/image.jpg")
        transformed_image = transformer.elastic_transform(original_image)
        ```
        """
        displacement_val = tf.random.normal([2, 3, 3]) * 5
        x_deformed = etf.deform_grid(
            inputs, displacement_val, axis=(0, 1), order=3)
        return x_deformed


class DistortImageLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Distorts the input image based on specified parameters.

        Parameters:
        - image (tf.Tensor): Input image as a TensorFlow tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Distorted image as a TensorFlow tensor.
        """

        if inputs.shape[-1] == 4:
            # Workaround for a bug in shear_x where alpha causes errors
            channel1, channel2, channel3, alpha = tf.split(inputs, 4, axis=2)
            image = tf.concat([channel1, channel2, channel3], axis=2)

            # Apply random JPEG quality to the image
            image = tf.image.random_jpeg_quality(image, 50, 100)

            # Split the channels again after distortion
            channel1, channel2, channel3 = tf.split(image, 3, axis=2)

            # Concatenate the channels along with alpha channel
            image = tf.concat([channel1, channel2, channel3, alpha], axis=2)

            # Clean up variables
            del channel1, channel2, channel3, alpha
        else:
            # Apply random JPEG quality to the image
            image = tf.image.random_jpeg_quality(inputs, 20, 100)

        return image


class RandomCropLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Applies random crop to the input image based on specified parameters.

        Parameters:
        - image (tf.Tensor): Input image as a TensorFlow tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Image after applying random crop as a TensorFlow tensor.
        """
        # Get the channels from the input
        channels = inputs.shape[-1]

        # Generate random seed for stateless operation
        random_seed = (random.randint(0, 100000), random.randint(0, 1000000))

        # Generate a random crop factor between 0.6 and 1.0
        random_crop_factor = \
        tf.random.uniform(shape=[1], minval=0.6, maxval=1.0)[0]

        # Get the original width and height of the image
        original_width = tf.shape(inputs)[1]
        original_height = tf.cast(tf.shape(inputs)[0], tf.float32)

        # Calculate the crop height based on the random factor
        crop_height = tf.cast(random_crop_factor * original_height, tf.int32)

        # Define the crop size
        crop_size = (crop_height, original_width, channels)

        # Apply stateless random crop to the image
        image = tf.image.stateless_random_crop(inputs, crop_size, random_seed)

        return image


class BinarizeSauvolaLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Binarize the input tensor using Sauvola's adaptive thresholding.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Binarized image tensor.
        """
        window_size = 51

        # Get the channels from the input
        channels = inputs.shape[-1]

        # 1-channel images don't have to be changed
        if channels == 1:
            np_array = inputs.numpy()
        elif channels == 3:
            np_array = tf.image.rgb_to_grayscale(inputs).numpy()
        elif channels == 4:
            # Drop alpha channel
            np_array = tf.image.rgb_to_grayscale(inputs[:, :, :3]).numpy()
        else:
            raise NotImplementedError(
                "Unsupported number of channels. Supported values are 1, "
                "3, or 4.")
        sauvola_thresh = threshold_sauvola(np_array, window_size=window_size)
        binary_sauvola = (np_array > sauvola_thresh) * 1

        return tf.convert_to_tensor(binary_sauvola)


class BinarizeOtsuLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Binarize the input tensor using Otsu's thresholding.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Binarized image tensor.

        Raises
        ------
        NotImplementedError
            If the number of channels is not supported.
            Supported values are 1, 3, or 4.

        """
        # Get the channels from the input
        channels = inputs.shape[-1]

        # 1-channel images don't have to be changed
        if channels == 1:
            np_array = inputs.numpy()
        elif channels == 3:
            np_array = tf.image.rgb_to_grayscale(inputs).numpy()
        elif channels == 4:
            # Drop alpha channel
            np_array = tf.image.rgb_to_grayscale(inputs[:, :, :3]).numpy()
        else:
            raise NotImplementedError(
                "Unsupported number of channels. Supported values are 1, "
                "3, or 4.")
        otsu_threshold = threshold_otsu(np_array)

        return tf.convert_to_tensor((np_array > otsu_threshold) * 1)


class InvertImageLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Invert the pixel values of the input tensor.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Inverted image tensor.
        """
        # Get the channels from the input
        channels = inputs.shape[-1]

        if (str(inputs.numpy().dtype).startswith("uint") or
                str(inputs.numpy().dtype).startswith("int")):
            max_value = 255
        else:
            max_value = 1

        if channels == 4:
            channel1, channel2, channel3, alpha = tf.split(inputs, 4, axis=2)
            channel1 = tf.convert_to_tensor(max_value - channel1.numpy())
            channel2 = tf.convert_to_tensor(max_value - channel2.numpy())
            channel3 = tf.convert_to_tensor(max_value - channel3.numpy())

            return tf.concat([channel1, channel2, channel3, alpha], axis=2)

        else:
            return tf.convert_to_tensor(max_value - inputs.numpy())


class BlurImageLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply random Gaussian blur to the input tensor

        Parameters:
        - image (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Blurred image tensor.
        """
        blur_factor = round(random.uniform(0.1, 5), 1)
        return tfm.vision.augment.gaussian_filter2d(inputs,
                                                    filter_shape=(10, 10),
                                                    sigma=blur_factor)


class RandomWidthLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Randomly adjusts the width of the input image.

        Parameters:
        - image (tf.Tensor): Input image as a TensorFlow tensor.

        Returns:
        - tf.Tensor: Processed image as a TensorFlow tensor.
        """

        # Get the width and height of the input image
        image_width = tf.shape(inputs)[1]
        image_height = tf.shape(inputs)[0]

        # Generate a random width scaling factor between 0.75 and 1.25
        random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]

        # Scale the width of the image by the random factor
        random_width *= float(image_width)
        image_width = int(random_width)

        # Convert the image to float32 dtype
        image = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

        # Resize the image to the new width while maintaining the original height
        image = tf.image.resize(image, [image_height, image_width])

        return image

def inspect_augments(image, augment_model):
    fig,ax = plt.subplots(nrows=10, figsize=(16, 10))
    fig.suptitle('Image augments:', fontsize=16)
    for i in range(10):
      augmented_image = augment_model(image)
      ax[i].imshow(augmented_image, cmap='gray')
      ax[i].set_title(str([layer.name for layer in augment_model.layers]))
    plt.tight_layout()
    plt.show()

# Test img
img_path = ""
image = tf.io.read_file(img_path)
image = tf.image.decode_png(image, channels=4)

# Build the Keras Sequential model
data_augmentation_lw = tf.keras.Sequential([
    ShearXLayer(),
    ElasticTransformLayer(),
    # DistortImageLayer(),
    # RandomCropLayer(),
    # BinarizeSauvolaLayer(),
    # BinarizeOtsuLayer(),
    # InvertImageLayer(),
    # BlurImageLayer(),
    # RandomWidthLayer(),
])

inspect_augments(image, data_augmentation_lw)