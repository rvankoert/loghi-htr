# Imports

# > Third party dependencies
import tensorflow as tf

# > Standard library
import unittest
import sys


class ModelCreationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.append("./src")
        from Model import build_model_new17
        cls.build_model = staticmethod(build_model_new17)

    def test_default_parameters(self):
        img_size = (32, 32, 3)
        number_characters = 26
        model = self.build_model(img_size, number_characters)
        print(model.summary())

        # Verify that a tf.keras.Model is created
        self.assertIsInstance(model, tf.keras.Model,
                              "Expected the model to be an instance of "
                              "tf.keras.Model")

        # Expect None for width, but use the other dimensions from img_size
        expected_input_shape = (None, None, img_size[1], img_size[2])

        # Verify the input and output shapes are as expected
        self.assertEqual(model.input_shape, expected_input_shape,
                         f"Expected input shape {expected_input_shape} but got"
                         f" {model.input_shape}")
        expected_output_shape = (None, number_characters + 1)
        self.assertEqual(model.output_shape[-1], expected_output_shape[-1],
                         f"Expected output shape {expected_output_shape[-1]} "
                         f"but got {model.output_shape[-1]}")

    def test_mask_usage(self):
        img_size = (32, 32, 3)
        number_characters = 26

        # Verify output shape without mask
        model_without_mask = self.build_model(
            img_size, number_characters, use_mask=False)
        self.assertEqual(
            model_without_mask.output_shape[-1], number_characters + 1,
            f"Expected output shape without mask to be {number_characters + 1}"
            f" but got {model_without_mask.output_shape[-1]}")

        # Verify output shape with mask
        model_with_mask = self.build_model(
            img_size, number_characters, use_mask=True)
        self.assertEqual(
            model_with_mask.output_shape[-1], number_characters + 2,
            f"Expected output shape with mask to be {number_characters + 2} "
            f"but got {model_with_mask.output_shape[-1]}")

    def test_gru_vs_lstm(self):
        img_size = (32, 32, 3)
        number_characters = 26

        model_with_gru = self.build_model(
            img_size, number_characters, use_gru=True)
        gru_layers = [layer for layer in model_with_gru.layers
                      if isinstance(layer, tf.keras.layers.Bidirectional)
                      and isinstance(layer.layer, tf.keras.layers.GRU)]
        self.assertTrue(len(gru_layers) > 0,
                        "Expected GRU layers wrapped with Bidirectional but "
                        "found none.")

        model_with_lstm = self.build_model(
            img_size, number_characters, use_gru=False)
        lstm_layers = [layer for layer in model_with_lstm.layers
                       if isinstance(layer, tf.keras.layers.Bidirectional)
                       and isinstance(layer.layer, tf.keras.layers.LSTM)]
        self.assertTrue(len(lstm_layers) > 0,
                        "Expected LSTM layers wrapped with Bidirectional but "
                        "found none.")

    # ETC
    # TODO: create and adjust tests for VGSL spec


if __name__ == "__main__":
    unittest.main()
