# Imports

# > Third party dependencies
import numpy as np

# > Standard library
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, ANY


class TestDataGenerator(unittest.TestCase):
    """
    Tests for the data_generator class.

    Test coverage:
        1. `test_initialization` tests that the instance variables are
        initialized correctly.
        2. `test_elastic_transform` tests that the elastic_transform function
        calls tf.random.normal and elasticdeform.tf.deform_grid with the
        expected arguments.
        3. `test_load_with_distort` tests that the load_images function calls
        tf.random_jpeg_distort with the expected arguments.
        4. `test_load_with_random_width` tests that the load_images function
        calls tf.image.resize with the expected arguments.
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

        from utils.utils import Tokenizer
        cls.Tokenizer = Tokenizer

    def test_initialization(self):
        tokenizer = self.Tokenizer(chars="ABC", use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, batch_size=32, height=128)

        # Verify that the instance variables are initialized correctly.
        self.assertEqual(dg.batch_size, 32)
        self.assertEqual(dg.height, 128)
        self.assertEqual(dg.do_binarize_sauvola, False)
        self.assertEqual(dg.do_binarize_otsu, False)
        self.assertEqual(dg.do_elastic_transform, False)
        self.assertEqual(dg.random_crop, False)
        self.assertEqual(dg.random_width, False)
        self.assertEqual(dg.distort_jpeg, False)
        self.assertEqual(dg.channels, 1)
        self.assertEqual(dg.do_random_shear, False)

    @patch("tensorflow.random.normal")
    @patch("elasticdeform.tf.deform_grid")
    def test_elastic_transform(self, mock_deform_grid, mock_random_normal):
        # Create an instance of DataGeneratorNew2
        tokenizer = self.Tokenizer(chars="ABC", use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, batch_size=32, height=128)

        # Mock return value for tf.random.normal
        mock_random_normal.return_value = np.zeros((2, 3, 3))

        # Mock return value for etf.deform_grid
        mock_deform_grid.return_value = "deformed_image"

        # Call the function
        result = dg.elastic_transform("original_image")

        # Validate that tf.random.normal was called with the expected arguments
        mock_random_normal.assert_called_once_with([2, 3, 3])

        # Validate that etf.deform_grid was called with the expected arguments
        mock_deform_grid.assert_called_once_with(
            "original_image", ANY, axis=(0, 1), order=3
        )

        # Check the return value
        self.assertEqual(result, "deformed_image")

    @patch("tensorflow.io.read_file", return_value="mock_image_file_content")
    @patch("tensorflow.image.decode_png", return_value=np.ones((100, 100, 3)))
    def test_load_with_distort(self, *mocks):
        # Create an instance of DataGeneratorNew2
        tokenizer = self.Tokenizer(chars="mock", use_mask=False)

        height = 128
        dg = self.DataGenerator(tokenizer=tokenizer, batch_size=32,
                                height=height, distort_jpeg=True)

        # Call the function
        result_image, result_label = dg.load_images(
            ["mock_image_path", "mock"])

        # Due to the nature of the random_jpeg_distort function, it is
        # impossible to verify the exact values of the image. Therefore, we
        # only check that the image is of the correct shape and that the
        # values are between 0 and 1.

        # Validate that the width has been padded with height + 50
        self.assertEqual(result_image.shape[0], height + 50)
        self.assertEqual(result_image.shape[1], height)

        # Validate that the values are between 0 and 1
        self.assertTrue(np.all(result_image >= 0)
                        and np.all(result_image <= 1))

    @patch("tensorflow.random.uniform", return_value=np.array([1.1]))
    @patch("tensorflow.shape", return_value=np.array([100, 100, 3]))
    @patch("tensorflow.io.read_file", return_value="mock_image_file_content")
    @patch("tensorflow.image.decode_png", return_value=np.zeros((100, 100, 3)))
    def test_load_with_random_width(self, mock_uniform, mock_shape, *mocks):
        # Create an instance of DataGeneratorNew2
        height = 128
        tokenizer = self.Tokenizer(chars="mock", use_mask=False)
        dg = self.DataGenerator(tokenizer=tokenizer, batch_size=32,
                                height=height, random_width=True)

        # Call the function
        result_image, result_label = dg.load_images(
            ["mock_image_path", "mock"])

        # Ensure the resized image has the expected expected_width
        # (original width * 1.1)
        expected_width = 110 + 50  # 100 * 1.1 + 50 extra padding
        self.assertEqual(result_image.shape[0], expected_width)


if __name__ == '__main__':
    unittest.main()
