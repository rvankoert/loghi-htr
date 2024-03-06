# Imports

# > Standard library
import logging
from pathlib import Path
import sys
import unittest

# > Local Dependencies

# > Third party dependencies
import tensorflow as tf


class TestDataGenerator(unittest.TestCase):
    """
    Tests for the data_generator class.

    Test coverage:
        1. `test_initialization` tests that the instance variables are
        initialized correctly.
        2. `test_load_images` test shapes before and after pre-processing
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        # Add the src directory to the path
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

        from data.generator import DataGenerator
        cls.DataGenerator = DataGenerator

        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

    def test_initialization(self):
        tokenizer = self.Tokenizer(chars=["ABC"], use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, height=128,
                                augment_model=None)

        # Verify that the instance variables are initialized correctly.
        self.assertEqual(dg.height, 128)
        self.assertEqual(dg.channels, 1)

    def test_load_images(self):
        # Set up a mock image file and label
        image_path = "path/to/mock_image.png"
        label = "mock_label"
        image_info_tuple = (image_path, label)

        tokenizer = self.Tokenizer(chars=["ABC"], use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, height=64, channels=3,
                                augment_model=None)

        # Mock TensorFlow's file reading and decoding operations
        with unittest.mock.patch.object(tf.io, 'read_file',
                                        return_value=tf.constant("mock_data")):
            with unittest.mock.patch.object(tf.image, 'decode_png',
                                            return_value=tf.ones([100, 100, 3])
                                            ):
                preprocessed_image, encoded_label = dg.load_images(
                    image_info_tuple)

                # Assert the shape of the preprocessed image
                self.assertEqual(preprocessed_image.shape, (304, 64, 3))
                self.assertIsInstance(preprocessed_image, tf.Tensor)
                self.assertIsInstance(encoded_label, tf.Tensor)


if __name__ == '__main__':
    unittest.main()
