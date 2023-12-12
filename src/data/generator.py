# Imports

# > Standard Library
import random
from typing import Tuple

# > Local dependencies

# > Third party libraries
import cv2
import tensorflow as tf
import elasticdeform.tf as etf
# import tensorflow_addons as tfa
from skimage.filters import threshold_otsu, threshold_sauvola
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 utils,
                 batch_size,
                 height=64,
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 do_elastic_transform=False,
                 random_crop=False,
                 random_width=False,
                 distort_jpeg=False,
                 channels=1,
                 do_random_shear=False,
                 do_blur=False,
                 do_invert=False
                 ):
        print(height)

        self.batch_size = batch_size
        self.do_binarize_sauvola = do_binarize_sauvola
        self.do_binarize_otsu = do_binarize_otsu
        self.do_elastic_transform = do_elastic_transform
        self.random_crop = random_crop
        self.random_width = random_width
        self.distort_jpeg = distort_jpeg
        self.utils = utils
        self.height = height
        self.channels = channels
        self.do_random_shear = do_random_shear
        self.do_blur = do_blur
        self.do_invert = do_invert

    def elastic_transform(self, original: np.ndarray) -> np.ndarray:
        """
        Apply elastic transformation to an image.

        Parameters
        ----------
        - original (numpy.ndarray): The original image to be transformed.

        Returns
        -------
        - numpy.ndarray: The image after elastic transformation.

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
            original, displacement_val, axis=(0, 1), order=3)
        return x_deformed

    def binarize_sauvola(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Binarize the input tensor using Sauvola's adaptive thresholding.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Binarized image tensor.
        """
        np_array = tensor.numpy()
        window_size = 51

        sauvola_thresh = threshold_sauvola(np_array, window_size=window_size)
        binary_sauvola = (np_array > sauvola_thresh) * 1

        return tf.convert_to_tensor(binary_sauvola)

    def binarize_otsu(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Binarize the input tensor using Otsu's thresholding.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Binarized image tensor.
        """
        np_array = tensor.numpy()

        np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)

        otsu_threshold = threshold_otsu(np_array)

        return tf.convert_to_tensor((np_array > otsu_threshold) * 1)

    def invert(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Invert the pixel values of the input tensor.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Inverted image tensor.
        """
        if str(tensor.numpy().dtype).startswith("uint") or str(tensor.numpy().dtype).startswith("int"):
            max_value = 255
        else:
            max_value = 1

        if self.channels == 4:
            channel1, channel2, channel3, alpha = tf.split(tensor, 4, axis=2)
            channel1 = tf.convert_to_tensor(max_value - channel1.numpy())
            channel2 = tf.convert_to_tensor(max_value - channel2.numpy())
            channel3 = tf.convert_to_tensor(max_value - channel3.numpy())

            return tf.concat([channel1, channel2, channel3, alpha], axis=2)

        else:
            return tf.convert_to_tensor(max_value - tensor.numpy())

    def blur(self, image: tf.Tensor) -> np.ndarray:
        """
        Apply Gaussian blur to the input tensor.

        Parameters:
        - image (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Blurred image tensor.
        """
        kernel_size = (11, 11)
        sigma_x = 0.0
        sigma_y = 20.0
        blurred_image = cv2.GaussianBlur(image.numpy(), kernel_size, sigma_x, sigma_y)
        return blurred_image
        # return tfa.image.gaussian_filter2d(tensor, sigma=[0.0, 20.0], filter_shape=(10, 10))

    def shear_x(self, image: np.ndarray, shear_factor: float) -> np.ndarray:
        """
        Apply shear transformation along the x-axis to the input image.

        Parameters
        ----------
        - image (numpy.ndarray): Input image, can be 3-channel (RGB) or
          4-channel (RGBA).
        - shear_factor (float): Shear factor to apply along the x-axis.

        Returns
        -------
        - numpy.ndarray: Sheared image.

        Raises
        ------
        - ValueError: If the input image has an unsupported number of channels.

        Example
        -------
        >>> input_image = np.random.rand(256, 256, 3)
        >>> sheared_result = self.shear_x(input_image, 0.5)
        """

        # Get image shapes
        rows, cols = image.shape[:2]

        # Define the shear matrix
        shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])

        # Split up channels from alpha (4-channel images)
        if self.channels == 4:
            image_rgb, image_alpha = tf.split(image, [3, 1], axis=-1)
        else:
            image_rgb = image

        # Apply the shear transformation
        sheared_image = cv2.warpAffine(
            image_rgb.numpy(), shear_matrix, (cols, rows))

        # Add back original alpha channel for (4-channel images)
        if self.channels == 4:
            repl_image = tf.concat([sheared_image, image_alpha], axis=-1)
        else:
            repl_image = sheared_image

        return repl_image

    def load_images(self, image_path: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess images.

        Parameters
        ----------
        - image_path (tuple): Tuple containing the file path (string) and label
          (string) of the image.

        Returns
        -------
        - Tuple: A tuple containing the preprocessed image (numpy.ndarray) and
          encoded label (numpy.ndarray).

        Raises
        ------
        - ValueError: If the number of channels is not 1, 3, or 4.

        Notes
        -----
        - This function uses TensorFlow operations to read, decode, and
          preprocess images.
        - Preprocessing steps include resizing, channel manipulation,
          distortion (if specified), elastic transform, cropping, shearing, and
          label encoding.

        Example:
        ```python
        loader = ImageLoader()
        image_path = ("/path/to/image.png", "label")
        preprocessed_image, encoded_label = loader.load_images(image_path)
        ```
        """
        image = tf.io.read_file(image_path[0])
        try:
            image = tf.image.decode_png(image, channels=self.channels)
        except ValueError:
            print("Invalid number of channels. "
                  "Supported values are 1, 3, or 4.")
        image = tf.image.resize(
            image, (self.height, 99999), preserve_aspect_ratio=True) / 255.0

        if self.distort_jpeg:
            if self.channels == 4:
                # crappy workaround for bug in shear_x where alpha causes
                # errors
                channel1, channel2, channel3, alpha = tf.split(
                    image, 4, axis=2)
                image = tf.concat([channel1, channel2, channel3], axis=2)
                image = tf.image.random_jpeg_quality(image, 50, 100)
                channel1, channel2, channel3 = tf.split(image, 3, axis=2)
                image = tf.concat(
                    [channel1, channel2, channel3, alpha], axis=2)
                del channel1, channel2, channel3, alpha
            else:
                image = tf.image.random_jpeg_quality(image, 20, 100)

        image_width = tf.shape(image)[1]
        image_height = tf.shape(image)[0]
        if self.do_elastic_transform:
            image = self.elastic_transform(image)

        if self.random_crop:
            random_seed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(
                shape=[1], minval=0.6, maxval=1.0)[0]
            original_width = tf.shape(image)[1]
            original_height = tf.cast(tf.shape(image)[0], tf.float32)
            crop_height = tf.cast(random_crop * original_height, tf.int32)
            crop_size = (crop_height, original_width, self.channels)

            image = tf.image.stateless_random_crop(
                image, crop_size, random_seed)
            image_width = tf.shape(image)[1]
            image_height = tf.shape(image)[0]

        if self.random_width:
            random_width = tf.random.uniform(
                shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            image_width = int(random_width)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, [image_height, image_width])

        image = tf.image.resize_with_pad(image, self.height, image_width+50)

        if self.do_random_shear:
            image = tf.image.resize_with_pad(
                image, self.height, image_width + 64 + 50)
            random_shear = tf.random.uniform(
                shape=[1], minval=-1.0, maxval=1.0)[0]
            if self.channels == 4:
                image = self.shear_x(image, random_shear * -1)
            elif self.channels == 3:
                image = self.shear_x(image, random_shear * -1)
            elif self.channels == 1:
                image = self.shear_x(image, random_shear * -1)
                image = np.expand_dims(image, axis=-1)
            else:
                raise NotImplementedError(
                    "Unsupported number of channels. Supported values are 1, "
                    "3, or 4.")

        if self.do_binarize_sauvola:
            image = self.binarize_sauvola(image)

        if self.do_binarize_otsu:
            image = self.binarize_otsu(image)

        if self.do_blur:
            image = self.blur(image)

        if self.do_invert:
            image = self.invert(image)

        label = image_path[1]
        encoded_label = self.utils.char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8"))

        label_counter = 0
        last_char = None

        for char in encoded_label:
            label_counter += 1
            if char == last_char:
                label_counter += 1
            last_char = char
        label_width = label_counter
        if image_width < label_width*16:
            image_width = label_width * 16
            image = tf.image.resize_with_pad(image, self.height, image_width)
        image = 0.5 - image
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, encoded_label
