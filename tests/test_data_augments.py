# Imports

# > Third party dependencies

# > Local dependencies
from data.augment_layers import *

# > Standard library
import unittest


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
    """

    def test_shear_x_layer(self):
        # Test ShearXLayer to ensure it applies a shear transformation
        layer = ShearXLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_elastic_transform_layer(self):
        # Test ElasticTransformLayer for elastic transformation
        layer = ElasticTransformLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_distort_image_layer(self):
        # Test DistortImageLayer for JPEG distortion
        layer = DistortImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_random_vertical_crop_layer(self):
        # Test RandomVerticalCropLayer for random cropping
        layer = RandomVerticalCropLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                # Height can be different from input other dims remain the same
                self.assertEqual(input_tensor.shape[0], output_tensor.shape[0])
                self.assertEqual(input_tensor.shape[2], output_tensor.shape[2])
                self.assertEqual(input_tensor.shape[3], output_tensor.shape[3])

    def test_random_width_layer(self):
        # Test RandomWidthLayer for width adjustment
        layer = RandomWidthLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_binarize_otsu(self):
        # Test BinarizeLayer for otsu adjustments
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='otsu', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_binarize_sauvola(self):
        # Test BinarizeLayer for sauvola adjustments
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                layer = BinarizeLayer(method='sauvola', channels=channels)
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_blur(self):
        # Test BlurImageLayer for blurring
        layer = BlurImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)

    def test_invert(self):
        # Test InvertImageLayer for pixel inversion
        layer = InvertImageLayer()
        for channels in [1, 3, 4]:
            with self.subTest(channels=channels):
                input_tensor = tf.random.uniform(shape=[1, 256, 256, channels])
                output_tensor = layer(input_tensor)
                self.assertEqual(input_tensor.shape, output_tensor.shape)


if __name__ == "__main__":
    unittest.main()
