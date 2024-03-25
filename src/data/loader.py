# Imports

# > Standard Library
from typing import Tuple

# > Third party libraries
import tensorflow as tf

# > Local dependencies
from utils.text import Tokenizer


class DataLoader:
    def __init__(self,
                 tokenizer: Tokenizer,
                 augment_model: tf.keras.Sequential,
                 height: int = 64,
                 channels: int = 1,
                 is_training: bool = False):
        """
        Initializes the DataLoader.

        Parameters
        ----------
        tokenizer: Tokenizer
            The tokenizer used for encoding labels.
        augment_model: tf.keras.Sequential
            The model used for data augmentation.
        height : int, optional
            The height of the preprocessed image (default is 64).
        channels : int, optional
            The number of channels in the image (default is 1).
        is_training : bool, optional
            Indicates whether the DataLoader is used for training (default is
            False).
        """

        self.tokenizer = tokenizer
        self.augment_model = augment_model
        self.height = height
        self.channels = channels
        self.is_training = is_training

    def load_images(self,
                    image_info_tuple: Tuple[str, str, str]) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Loads, preprocesses a single image, and encodes its label.

        Parameters
        ----------
        image_info_tuple : Tuple[str, str, str]
            A tuple containing image path, label, and sample weight.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing the preprocessed image, encoded label, and
            sample weight.
        """

        # Load and preprocess the image
        image, original_width = self._load_and_preprocess_image(
            image_info_tuple[0])

        # Encode the label
        encoded_label = self.tokenizer(image_info_tuple[1])

        # Ensure the image width is sufficient for CTC decoding
        image = self._ensure_width_for_ctc(
            image, encoded_label, original_width)

        # Center the image values around 0.5
        image = 0.5 - image

        # Transpose the image
        image = tf.transpose(image, perm=[1, 0, 2])

        # Get the sample weight
        sample_weight = tf.strings.to_number(image_info_tuple[2])

        return image, encoded_label, sample_weight

    def _load_and_preprocess_image(self, image_path: str) \
            -> Tuple[tf.Tensor, int]:
        """
        Loads and preprocesses a single image.

        Parameters
        ----------
        image_path: str
            The path to the image file.

        Returns
        -------
        Tuple[tf.Tensor, int]
            A tuple containing the preprocessed image and the original width
            of the image.

        Raises
        ------
        ValueError
            If the number of channels is not 1, 3, or 4.
        """

        # 1. Load the Image
        image = tf.io.read_file(image_path)

        try:
            image = tf.image.decode_image(image, channels=self.channels,
                                          expand_animations=False)
        except ValueError as e:
            raise ValueError("Invalid number of channels. "
                             "Expected 1, 3, or 4.") from e

        # 2. Resize the Image and Normalize Pixel Values to [0, 1]
        image = tf.image.resize(image, (self.height, 99999),
                                preserve_aspect_ratio=True) / 255.0

        original_width = tf.shape(image)[1]

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

        return tf.cast(image[0], tf.float32), original_width

    def _ensure_width_for_ctc(self,
                              image: tf.Tensor,
                              encoded_label: tf.Tensor,
                              original_width: int) -> tf.Tensor:
        """
        Resizes the image if necessary to accommodate the encoded label during
        CTC decoding.

        Parameters
        ----------
        image : tf.Tensor
            The preprocessed image tensor.
        encoded_label : tf.Tensor
            The encoded label.
        original_width : int
            The original width of the image.

        Returns
        -------
        tf.Tensor
            The resized image tensor.
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

        if original_width < required_width:
            image = tf.image.resize_with_pad(
                image, self.height, required_width)

        return image
