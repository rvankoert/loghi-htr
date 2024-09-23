# Imports

# > Standard Library
from typing import Tuple, Optional

# > Third party libraries
import tensorflow as tf

# > Local dependencies
from utils.text import Tokenizer


class DataLoader:
    def __init__(self,
                 tokenizer: Tokenizer,
                 augmentations: tf.keras.Sequential,
                 height: int = 64,
                 channels: int = 1,
                 is_training: bool = False):
        """
        Initializes the DataLoader.

        Parameters
        ----------
        tokenizer: Tokenizer
            The tokenizer used for encoding labels.
        augmentations: tf.keras.Sequential
            The data augmentation layers.
        height : int, optional
            The height of the preprocessed image (default is 64).
        channels : int, optional
            The number of channels in the image (default is 1).
        is_training : bool, optional
            Indicates whether the DataLoader is used for training (default is False).
        """
        self.tokenizer = tokenizer
        self.augmentations = augmentations
        self.height = height
        self.channels = channels
        self.is_training = is_training

    @tf.function
    def load_image(self, image_path: tf.Tensor) -> tf.Tensor:
        """
        Loads and preprocesses a single image.

        Parameters
        ----------
        image_path: tf.Tensor
            The path to the image file.

        Returns
        -------
        tf.Tensor
            The preprocessed image tensor.
        """
        # Load the image
        image_content = tf.io.read_file(image_path)

        # Decode the image (assuming images are in PNG format)
        image = tf.io.decode_image(
            image_content, channels=self.channels, expand_animations=False)

        # Resize and normalize the image
        image = tf.image.resize(
            image, [self.height, tf.constant(99999, dtype=tf.int32)], preserve_aspect_ratio=True)
        image = tf.cast(image, tf.float32) / 255.0

        # Add batch dimension (required for augmentation model)
        image = tf.expand_dims(image, 0)

        for layer in self.augmentations.layers:
            # Custom layer handling (assuming 'extra_resize_with_pad'
            # remains)
            if layer.name == "extra_resize_with_pad":
                image = layer(image, training=True)
            else:
                image = layer(image, training=self.is_training)

        # Remove the batch dimension
        image = tf.squeeze(image, axis=0)

        # Center the image values around 0.5
        image = image - 0.5

        # Transpose the image
        image = tf.transpose(image, perm=[1, 0, 2])

        return image

    @tf.function
    def process_sample(self, inputs) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Processes a single sample consisting of an image path, label, and sample weight.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing the image path, label, and sample weight.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing the preprocessed image, encoded label, and sample weight.
        """

        # Load and preprocess the image
        image = self.load_image(inputs[0])

        # Encode the label
        encoded_label = self.tokenizer(inputs[1])

        # Ensure the image width is sufficient for CTC decoding
        image = self._ensure_width_for_ctc(image, encoded_label)

        sample_weight = tf.strings.to_number(
            inputs[2], out_type=tf.float32)

        return image, encoded_label, sample_weight

    @tf.function
    def _ensure_width_for_ctc(self, image: tf.Tensor, encoded_label: tf.Tensor) -> tf.Tensor:
        """
        Ensures that the image width is sufficient for CTC decoding.

        Parameters
        ----------
        image : tf.Tensor
            The preprocessed image tensor.
        encoded_label : tf.Tensor
            The encoded label.

        Returns
        -------
        tf.Tensor
            The resized image tensor.
        """
        # Calculate the required width for the image
        required_width = tf.shape(encoded_label)[
            0] + tf.reduce_sum(tf.cast(tf.equal(encoded_label[:-1], encoded_label[1:]), tf.int32))

        # Convert to pixels
        pixels_per_column = 16
        required_width *= pixels_per_column

        # Pad the image if necessary
        current_width = tf.shape(image)[0]
        width_diff = required_width - current_width
        width_diff = tf.maximum(width_diff, 0)

        if width_diff > 0:
            padding = [[0, width_diff], [0, 0], [0, 0]]
            image = tf.pad(image, padding, mode='CONSTANT')

        return image
