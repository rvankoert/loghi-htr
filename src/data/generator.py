# Imports

# > Standard Library
from typing import Tuple

# > Local dependencies
from augments import *

# > Third party libraries
import tensorflow as tf
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

    def load_images(self, image_info_tuple: Tuple[str, str]) -> (
            Tuple)[np.ndarray, np.ndarray]:
        """
        Load and preprocess images.

        Parameters
        ----------
        - image_info_tuple (tuple): Tuple containing the file path (string)
        and label(string) of the image.

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
        image_info_tuple = ("/path/to/image.png", "label")
        preprocessed_image, encoded_label = loader.load_images(image_info_tuple)
        ```
        """
        image = tf.io.read_file(image_info_tuple[0])
        try:
            image = tf.image.decode_png(image, channels=self.channels)
        except ValueError:
            print("Invalid number of channels. "
                  "Supported values are 1, 3, or 4.")
        image = tf.image.resize(
            image, (self.height, 99999), preserve_aspect_ratio=True) / 255.0

        if self.distort_jpeg:
            image = distort_image(image, self.channels)

        image_width = tf.shape(image)[1]
        if self.do_elastic_transform:
            image = elastic_transform(image)

        if self.random_crop:
            image = random_crop(image, self.channels)

        if self.random_width:
            image = random_width(image)

        image = tf.image.resize_with_pad(image,
                                         self.height, tf.shape(image)[1] + 50)

        if self.do_random_shear:
            # Add padding to make sure that shear does not "fall off" image
            image = tf.image.resize_with_pad(
                image, self.height, image_width + 64 + 50)
            if self.channels in [3, 4]:
                image = shear_x(image)
            elif self.channels == 1:
                image = shear_x(image)
                image = np.expand_dims(image, axis=-1)
            else:
                raise NotImplementedError(
                    "Unsupported number of channels. Supported values are 1, "
                    "3, or 4.")

        if self.do_binarize_sauvola:
            image = binarize_sauvola(image, self.channels)

        if self.do_binarize_otsu:
            image = binarize_otsu(image, self.channels)

        if self.do_blur:
            image = blur_image(image)

        if self.do_invert:
            image = invert_image(image, self.channels)

        label = image_info_tuple[1]

        # Convert label to numeric representation
        encoded_label = self.utils.char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8"))

        # Calculate label width
        label_width = tf.reduce_sum(
            tf.cast(
                tf.not_equal(
                    encoded_label[1:],
                    encoded_label[:-1]),
                tf.int32)
        ) + 1

        # Update image width if needed
        image_width = tf.maximum(image_width, label_width * 16)
        image = tf.image.resize_with_pad(image, self.height, image_width)

        # Misc operations
        image = 0.5 - image
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, encoded_label
