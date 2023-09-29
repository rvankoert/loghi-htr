# Imports

# > Third party dependencies
import tensorflow as tf
from tensorflow.keras import layers

# > Standard library
import logging
import unittest
from pathlib import Path
import sys


class TestModelToVGSL(unittest.TestCase):
    """
    Test for the model_to_vgsl function in the VGSLModelGenerator class.

    The model_to_vgsl function takes a Keras model and returns a VGSL
    specification string. This test suite checks that the function returns the
    correct VGSL specification string for a given Keras model.

    Test coverage:
        1. `test_input_layer_to_vgsl`: Test the conversion of an input layer
        2. `test_input_layer_error`: Test the conversion of an input layer with
            an invalid input shape
        3. `test_output_layer_to_vgsl`: Test the conversion of an output layer
        4. `test_conv2d_layer_to_vgsl`: Test the conversion of a Conv2D layer
        5. `test_dense_layer_to_vgsl`: Test the conversion of a Dense layer
        6. `test_lstm_layer_to_vgsl`: Test the conversion of an LSTM layer
        7. `test_gru_layer_to_vgsl`: Test the conversion of a GRU layer
        8. `test_bidirectional_layer_to_vgsl`: Test the conversion of a
            Bidirectional layer
        9. `test_batch_normalization_to_vgsl`: Test the conversion of a
            BatchNormalization layer
        10. `test_max_pooling_2d_to_vgsl`: Test the conversion of a
            MaxPooling2D layer
        11. `test_avg_pooling_2d_to_vgsl`: Test the conversion of an
            AveragePooling2D layer
        12. `test_dropout_to_vgsl`: Test the conversion of a Dropout layer
        13. `test_residual_block_to_vgsl`: Test the conversion of a Residual
            Block layer
        14. `test_functional_combination_to_vgsl`: Test the conversion of a
            model defined using the functional API
        15. `test_sequential_combination_to_vgsl`: Test the conversion of a
            model defined using the sequential API
    """

    @classmethod
    def setUpClass(cls):
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        from vgsl_model_generator import VGSLModelGenerator
        cls.VGSLModelGenerator = VGSLModelGenerator

        from custom_layers import ResidualBlock
        cls.ResidualBlock = ResidualBlock

    def test_input_layer_to_vgsl(self):
        # Basic input layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dense(10, activation='softmax')
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 O1s10")

    def test_input_layer_error(self):
        # Basic input layer
        model = tf.keras.Sequential([
            layers.Input(shape=(64, 1)),
            layers.Dense(10, activation='softmax')
        ])

        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator.model_to_vgsl(model)

        self.assertEqual(
            str(context.exception), "Invalid input shape (None, 64, 1). Input "
            "shape must be of the form (None, height, width, channels).")

    def test_output_layer_to_vgsl(self):
        # Basic output layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dense(10, activation='softmax')
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 O1s10")

        # Alternative output layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dense(3, activation='sigmoid')
        ])
        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 O1m3")

    def test_conv2d_layer_to_vgsl(self):
        # Basic conv2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Cr3,3,32")

        # Alternative conv2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Conv2D(128, (2, 2), activation='sigmoid'),
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Cm2,2,128")

    def test_dense_layer_to_vgsl(self):
        # Basic dense layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Fr64 O1s10")

        # Alternative dense layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dense(16, activation='tanh'),
            layers.Dense(10, activation='softmax')
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Ft16 O1s10")

    def test_lstm_layer_to_vgsl(self):
        # Basic LSTM layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.LSTM(64)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Rc Lf64")

        # Alternative LSTM layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.LSTM(128, return_sequences=True, go_backwards=True),
            layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1),
            layers.LSTM(64, dropout=0.25, recurrent_dropout=0.3)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 Rc Lrs128 Lfs128,Rd10 Lf64,D25,Rd30")

    def test_gru_layer_to_vgsl(self):
        # Basic GRU layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.GRU(64)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Rc Gf64")

        # Alternative GRU layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.GRU(128, return_sequences=True, go_backwards=True),
            layers.GRU(128, return_sequences=True, recurrent_dropout=0.1),
            layers.GRU(64, dropout=0.25, recurrent_dropout=0.3)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 Rc Grs128 Gfs128,Rd10 Gf64,D25,Rd30")

    def test_bidirectional_layer_to_vgsl(self):
        # Basic bidirectional layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
            layers.Bidirectional(layers.GRU(256)),
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Rc Bl256 Bg256")

        # Alternative bidirectional layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Reshape((-1, 64)),
            layers.Bidirectional(layers.LSTM(256,
                                             return_sequences=True,
                                             dropout=0.25,
                                             recurrent_dropout=0.3)),
            layers.Bidirectional(layers.GRU(256, dropout=0.25))
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 Rc Bl256,D25,Rd30 Bg256,D25")

    def test_batch_normalization_to_vgsl(self):
        # Basic batch normalization layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.BatchNormalization()
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Bn")

    def test_max_pooling_2d_to_vgsl(self):
        # Basic max pooling 2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Mp2,2,2,2")

        # Alternative max pooling 2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.MaxPooling2D(pool_size=(3, 1), strides=(1, 1))
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Mp3,1,1,1")

    def test_avg_pooling_2d_to_vgsl(self):
        # Basic average pooling 2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.AveragePooling2D(pool_size=(2, 2))
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Ap2,2,2,2")

        # Alternative average pooling 2d layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.AveragePooling2D(pool_size=(3, 1), strides=(1, 1))
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 Ap3,1,1,1")

    def test_dropout_to_vgsl(self):
        # Basic dropout layer
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Dropout(0.25)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 D25")

    def test_residual_block_to_vgsl(self):
        # Basic residual block
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            self.ResidualBlock(32, 3, 3)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(vgsl_spec, "None,64,None,1 RB3,3,32")

        # Alternative residual block
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            self.ResidualBlock(16, 4, 2, downsample=True)
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 RBd4,2,16")

    def test_functional_combination_to_vgsl(self):
        # Define the original model using the functional API
        input_tensor = layers.Input(shape=(None, 64, 1))
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = self.ResidualBlock(32, 3, 3)(x)
        x = layers.Reshape((-1, 1024))(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.GRU(64, return_sequences=True, go_backwards=True)(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(x)
        x = layers.Dense(10, activation='softmax')(x)
        output_tensor = layers.Activation('linear')(x)
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 Cr3,3,32 Mp2,2,2,2 Ap2,2,1,1 Bn D25 "
                       "RB3,3,32 Rc Lfs64 Grs64 Bl256 Bg256 O1s10")

        # Generate a model from the VGSL string
        model_generator = self.VGSLModelGenerator(vgsl_spec)
        generated_model = model_generator.build()

        # Compare the original model with the generated model
        self.compare_model_configs(model, generated_model)

    def test_sequential_combination_to_vgsl(self):
        # Define the original model using the sequential API
        model = tf.keras.Sequential([
            layers.Input(shape=(None, 64, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            self.ResidualBlock(32, 3, 3),
            layers.Reshape((-1, 1024)),
            layers.LSTM(64, return_sequences=True),
            layers.GRU(64, return_sequences=True, go_backwards=True),
            layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
            layers.Bidirectional(layers.GRU(256, return_sequences=True)),
            layers.Dense(10, activation='softmax'),
            layers.Activation('linear')
        ])

        vgsl_spec = self.VGSLModelGenerator.model_to_vgsl(model)
        self.assertEqual(
            vgsl_spec, "None,64,None,1 Cr3,3,32 Mp2,2,2,2 Ap2,2,1,1 Bn D25 "
                       "RB3,3,32 Rc Lfs64 Grs64 Bl256 Bg256 O1s10")

        # Generate a model from the VGSL string
        model_generator = self.VGSLModelGenerator(vgsl_spec)
        generated_model = model_generator.build()

        # Compare the original model with the generated model
        self.compare_model_configs(model, generated_model)

    def compare_model_configs(self, model, generated_model):

        start_idx_original = 0
        start_idx_generated = 1 if not isinstance(
            model.layers[0], tf.keras.layers.InputLayer) else 0

        # Check number of layers
        if start_idx_generated == 1:
            self.assertEqual(len(model.layers) + 1, len(
                generated_model.layers), "Number of layers do not match")
        else:
            self.assertEqual(len(model.layers), len(
                generated_model.layers), "Number of layers do not match")

        # Loop through each layer and compare configurations
        for original_layer, generated_layer in \
                zip(model.layers[start_idx_original:],
                    generated_model.layers[start_idx_generated:]):
            original_config = original_layer.get_config().copy()
            generated_config = generated_layer.get_config().copy()

            # Remove the specified keys from both configurations
            keys_to_ignore = ['name', 'kernel_initializer',
                              'initializer', 'padding']

            # If the layer is Bidirectional, update the inner layer's config
            if isinstance(original_layer, tf.keras.layers.Bidirectional):
                for key in keys_to_ignore:
                    original_config['layer']['config'].pop(key, None)
                    generated_config['layer']['config'].pop(key, None)
            for key in keys_to_ignore:
                original_config.pop(key, None)
                generated_config.pop(key, None)

            self.assertEqual(original_config,
                             generated_config,
                             "Configuration mismatch in "
                             f"{type(original_layer).__name__}")


if __name__ == '__main__':
    unittest.main()
