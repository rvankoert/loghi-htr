# Imports

# > Third party dependencies
import tensorflow as tf

# > Standard library
import logging
import unittest
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

# Local dependencies
from data.augment_layers import (ShearXLayer,
                                 ElasticTransformLayer,  # noqa: E402
                                 DistortImageLayer, RandomVerticalCropLayer,
                                 RandomWidthLayer, BinarizeLayer,
                                 BlurImageLayer, InvertImageLayer)


class TestDataAugments(unittest.TestCase):
    """
    Tests for verifying functionality of image augmentation layers.

    Test coverage:
        1. `test_shear_x_layer`: Tests ShearXLayer for shear transformation
           and shape consistency.
        2. `test_elastic_transform_layer`: Tests ElasticTransformLayer for
           elastic transformations, checking shape consistency.
        3. `test_distort_image_layer`: Tests DistortImageLayer for JPEG-like
           distortions, verifying shape consistency.
        4. `test_random_vertical_crop_layer`: Tests RandomVerticalCropLayer
           for vertical cropping, checking dimension consistency.
        5. `test_random_width_layer`: Tests RandomWidthLayer for width
           adjustment, maintaining dimension consistency.
        6. `test_binarize_otsu`: Tests BinarizeLayer with Otsu method for
           binary thresholding, verifying shape consistency.
        7. `test_binarize_sauvola`: Tests BinarizeLayer with Sauvola method,
           ensuring shape consistency post-binarization.
        8. `test_blur`: Tests BlurImageLayer for applying blur and
           maintaining original shape.
        9. `test_invert`: Tests InvertImageLayer for color inversion and
           shape consistency.
        10.`test_augment_selection`: Tests a pre-set number of augments in
        succession
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.DEBUG,
        )

        cls.possible_layers = [ShearXLayer(), ElasticTransformLayer(),
                               DistortImageLayer(), RandomVerticalCropLayer(),
                               RandomWidthLayer(), BinarizeLayer('otsu'),
                               BinarizeLayer('sauvola'), BlurImageLayer(),
                               InvertImageLayer()]

    def test_shear_x_layer(self):
        # Test ShearXLayer to ensure it applies a shear transformation
        layer = ShearXLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_elastic_transform_layer(self):
        # Test ElasticTransformLayer for elastic transformation
        layer = ElasticTransformLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_distort_image_layer(self):
        # Test DistortImageLayer for JPEG distortion
        layer = DistortImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_random_vertical_crop_layer(self):
        # Test RandomVerticalCropLayer for random cropping
        layer = RandomVerticalCropLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)

                random_crop_factor = layer.crop_factor

                # Recalculate crop height based on input
                new_height = tf.cast(random_crop_factor *
                                     tf.cast(256, tf.float32), tf.int32)

                # Apply expected crop
                old_width = input_tensor.shape[2]
                input_tensor = tf.image.resize(input_tensor,
                                               [new_height, old_width])

                # Check if applied crop is the same as the output
                self.assertEqual(input_tensor.shape,
                                 output_tensor.shape)

    def test_random_width_layer(self):
        # Test RandomWidthLayer for width adjustment
        layer = RandomWidthLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                random_width_factor = layer.random_width_factor

                # Scale the width of the image by the random factor
                new_width = int(random_width_factor * float(256))

                # Apply expected width change
                old_height = input_tensor.shape[1]
                input_tensor = tf.image.resize(input_tensor,
                                               [old_height, new_width])

                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_binarize_otsu(self):
        # Test BinarizeLayer for otsu adjustments and channel check
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='otsu', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_binarize_sauvola(self):
        # Test BinarizeLayer for sauvola adjustments and channel check
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='sauvola', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_blur(self):
        # Test BlurImageLayer for blurring
        layer = BlurImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_invert(self):
        # Test InvertImageLayer for pixel inversion
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = InvertImageLayer(channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor, training=True)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_simple_augment_selection(self):
        # Test simple sequential image augments
        augment_selection_a = (self.possible_layers[0:3] +
                               [self.possible_layers[5]])

        # Test without changing image height / width
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                for layer in augment_selection_a:
                    # Check if BinarizeLayer so channels can be set correctly
                    if isinstance(layer, BinarizeLayer):
                        if layer.method == "otsu":
                            layer = BinarizeLayer("otsu",
                                                  channels=channels)
                        else:
                            layer = BinarizeLayer("sauvola",
                                                  channels=channels)

                    output_tensor = layer(input_tensor, training=True)
                    self.assertEqual(input_tensor.shape,
                                     output_tensor.shape)

    def test_hw_changing_augment_selection(self):
        # Test sequential image augments with changing image
        # height and width dimensions
        logging.debug("Testing sequential image augmentations:")
        augment_selection_b = self.possible_layers

        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(
                    shape=[1, 256, 256, channels])
                logging.debug("Initial input size = " + str(input_tensor.shape))
                for layer in augment_selection_b:
                    # Check if BinarizeLayer so channels can be set correctly
                    if isinstance(layer, BinarizeLayer):
                        if layer.method == "otsu":
                            layer = BinarizeLayer("otsu",
                                                  channels=channels)
                        else:
                            layer = BinarizeLayer("sauvola",
                                                  channels=channels)

                    # Apply augment to image
                    output_tensor = layer(input_tensor, training=True)

                    if layer.name == "random_vertical_crop_layer":
                        # Height of image reduces with random factor
                        crop_factor = layer.crop_factor

                        # Recalculate crop height based on input
                        new_height = tf.cast(crop_factor *
                                             tf.cast(256, tf.float32), tf.int32)

                        # Apply expected crop
                        old_width = input_tensor.shape[2]
                        input_tensor = tf.image.resize(input_tensor,
                                                       [new_height, old_width])

                        self.assertEqual(input_tensor.shape,
                                         output_tensor.shape)

                    elif layer.name == "random_width_layer":
                        # Width of image changes with random factor
                        random_width_factor = layer.random_width_factor

                        # Scale the width of the image by the random factor
                        new_width = int(random_width_factor * float(256))

                        # Apply expected width change
                        old_height = input_tensor.shape[1]
                        input_tensor = tf.image.resize(input_tensor,
                                                       [old_height, new_width])

                        self.assertEqual(input_tensor.shape,
                                         output_tensor.shape)

                    else:
                        self.assertEqual(input_tensor.shape,
                                         output_tensor.shape)

                    logging.debug("Dims after " + layer.name + " = " +
                                  str(input_tensor.shape))
        logging.debug("Final input dims: " + str(input_tensor.shape))


if __name__ == "__main__":
    unittest.main()
