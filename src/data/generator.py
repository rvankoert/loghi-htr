# Imports

# > Standard Library
import logging
from typing import Tuple

# > Local dependencies

# > Third party libraries
import tensorflow as tf
import numpy as np


class DataGenerator:
    def __init__(self,
                 tokenizer,
                 augment_model,
                 height=64,
                 channels=1,
                 is_training=False,
                 ):
        self.tokenizer = tokenizer
        self.augment_model = augment_model
        self.height = height
        self.channels = channels
        self.is_training = is_training

    def load_images(self, image_info_tuple: Tuple[str, str, str]) -> (
            Tuple)[np.ndarray, np.ndarray]:
        """
        Loads, preprocesses a single image, and encodes its label.

        Unpacks the tuple for readability.
        """

        # Load and preprocess the image
        image = self._load_and_preprocess_image(image_info_tuple[0])

        # Encode the label
        encoded_label = self.tokenizer(image_info_tuple[1])

        # Ensure the image width is sufficient for CTC decoding
        image = self._ensure_width_for_ctc(image, encoded_label)

        # Center the image values around 0.5
        image = 0.5 - image

        # Transpose the image
        image = tf.transpose(image, perm=[1, 0, 2])

        # Get the sample weight
        sample_weight = tf.strings.to_number(image_info_tuple[2])

        return image, encoded_label, sample_weight

    def _load_and_preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Loads and preprocesses a single image.

        Parameters
        ----------
        image_path: str
            The path to the image file.

        Returns
        -------
        tf.Tensor
            A preprocessed image tensor ready for training.
        """

        # 1. Load the Image
        image = tf.io.read_file(image_path)

        try:
            image = tf.image.decode_image(image, channels=self.channels,
                                          expand_animations=False)
        except ValueError:
            logging.error("Invalid number of channels. "
                          "Supported values are 1, 3, or 4.")

        # 2. Resize the Image and Normalize Pixel Values to [0, 1]
        image = tf.image.resize(image, (self.height, 99999),
                                preserve_aspect_ratio=True) / 255.0

        # 3. Apply Data Augmentations
        # Add batch dimension (required for augmentation model)
        image = tf.expand_dims(image, 0)

        for layer in self.augment_model.layers:
            # Custom layer handling (assuming 'extra_resize_with_pad'
            # remains)
            if layer.name == "extra_resize_with_pad":
                image = layer(image, training=True)
            else:
                image = layer(image, training=self.is_training)

        return tf.cast(image[0], tf.float32)

    def _ensure_width_for_ctc(self, image, encoded_label):
        """Resizes the image if necessary to accommodate the encoded label
        during CTC decoding.
        """

        # Calculate the required width for the image
        required_width = len(encoded_label)

        num_repetitions = 0
        last_char = None
        for char in encoded_label:
            if char == last_char:
                num_repetitions += 1
            last_char = char

        # Add repetitions
        required_width += num_repetitions

        # Convert to pixels
        pixels_per_column = 16
        required_width *= pixels_per_column

        # Mandatory cast to float32
        image = tf.cast(image, tf.float32)

        if tf.shape(image)[1] < required_width:
            image = tf.image.resize_with_pad(
                image, self.height, required_width)

        return image
