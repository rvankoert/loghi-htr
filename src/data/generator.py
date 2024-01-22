# Imports

# > Standard Library
import logging
from typing import Tuple

# > Local dependencies

# > Third party libraries
import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 tokenizer,
                 batch_size,
                 height=64,
                 channels=1,
                 augment_model=None,
                 is_training=True,
                 ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.height = height
        self.channels = channels
        self.augment_model = augment_model
        self.is_training = is_training

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
            logging.error("Invalid number of channels. "
                          "Supported values are 1, 3, or 4.")
        image = tf.image.resize(image, (self.height, 99999),
                                preserve_aspect_ratio=True) / 255.0

        # Add batch dimension
        image = tf.expand_dims(image, 0)

        # Apply augmentations
        if self.augment_model is not None:
            for layer in self.augment_model.layers:
                if not self.is_training and \
                        isinstance(layer, tf.keras.layers.Cropping2D):
                    continue
                image = layer(image, training=self.is_training)

        # Remove batch dimension
        image = image[0]

        image_width = tf.shape(image)[1]
        image = tf.image.resize_with_pad(image, self.height, image_width+50)

        label = image_info_tuple[1]
        encoded_label = self.tokenizer(label)

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
