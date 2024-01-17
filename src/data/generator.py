# Imports

# > Standard Library
from typing import Tuple

# > Local dependencies

# > Third party libraries
import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 utils,
                 batch_size,
                 height=64,
                 channels=1,
                 aug_binarize_sauvola=False,
                 aug_binarize_otsu=False,
                 aug_elastic_transform=False,
                 aug_random_crop=False,
                 aug_random_width=False,
                 aug_distort_jpeg=False,
                 aug_random_shear=False,
                 aug_blur=False,
                 aug_invert=False
                 ):
        self.utils = utils
        self.batch_size = batch_size
        self.height = height
        self.channels = channels
        self.aug_random_shear = aug_random_shear
        self.aug_blur = aug_blur
        self.aug_invert = aug_invert
        self.aug_binarize_sauvola = aug_binarize_sauvola
        self.aug_binarize_otsu = aug_binarize_otsu
        self.aug_elastic_transform = aug_elastic_transform
        self.aug_random_crop = aug_random_crop
        self.aug_random_width = aug_random_width
        self.aug_distort_jpeg = aug_distort_jpeg

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
        # Read in image info tuple
        image = tf.io.read_file(image_info_tuple[0])
        try:
            image = tf.image.decode_png(image, channels=self.channels)
        except ValueError:
            print("Invalid number of channels. "
                  "Supported values are 1, 3, or 4.")
        image = tf.image.resize(
            image, (self.height, 99999), preserve_aspect_ratio=True) / 255.0

        image_width = tf.shape(image)[1]

        # Start encoding the tabel part
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