# Imports

# > Third party dependencies
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations

# > Standard library
import logging
import unittest
import sys


class VGSLModelGeneratorTest(unittest.TestCase):
    """
    Tests for creating a new model.

    Test coverage:
        1. `test_create_simple_model`: Test model creation with a simple
            VGSL-spec string.
        2. `test_conv2d_layer`: Test model creation with a Conv2D layer.
        3. `test_maxpool_layer`: Test model creation with a MaxPooling2D layer.
        4. `test_avgpool_layer`: Test model creation with a AvgPool2D layer.
        5. `test_reshape_layer`: Test model creation with a Reshape layer.
        6. `test_fully_connected_layer`: Test model creation with a Fully
            connected layer.
        7. `test_lstm_layer`: Test model creation with a LSTM layer.
        8. `test_gru_layer`: Test model creation with a GRU layer.
        9. `test_bidirectional_layer`: Test model creation with a Bidirectional
            layer.
        10. `test_residual_block`: Test model creation with a Residual block.
        11. `test_dropout_layer`: Test model creation with a Dropout layer.
        12. `test_output_layer`: Test model creation with an Output layer.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.append("./src")

        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        from vgsl_model_generator import VGSLModelGenerator
        cls.VGSLModelGenerator = VGSLModelGenerator

        from custom_layers import CTCLayer, ResidualBlock
        cls.ResidualBlock = ResidualBlock
        cls.CTCLayer = CTCLayer

    def test_create_simple_model(self):
        # VGSL-spec string for a basic model with an input layer, a convolution
        # layer, and an output layer
        vgsl_spec_string = "None,64,None,1 Cr3,3,32 O1s10"

        # Instantiate the VGSLModelGenerator object
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)

        # Build the model
        model = model_generator.build()

        # Check if the model is not None
        self.assertIsNotNone(model)

        # Check if the model name is set to "custom_model" as the
        # vgsl_spec_string didn't start with "model"
        self.assertEqual(model_generator.model_name, "custom_model")

        # Check if the number of layers in the model is 4
        # (Input, Conv2D, Dense, Activation)
        self.assertEqual(len(model.layers), 4)

        # Check that each layer is of the correct type
        self.assertIsInstance(model.layers[0], layers.InputLayer)
        self.assertIsInstance(model.layers[1], layers.Conv2D)
        self.assertIsInstance(model.layers[2], layers.Dense)
        self.assertIsInstance(model.layers[3], layers.Activation)

    def test_conv2d_layer(self):
        vgsl_spec_string = "None,64,None,1 Cr3,3,32 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Check that the second layer is a Conv2D layer
        self.assertIsInstance(model.layers[1], layers.Conv2D)

        # Layer-specicific tests
        # Check that the Conv2D layer has the correct number of filters
        self.assertEqual(model.layers[1].filters, 32)

        # Check that the Conv2D layer has the correct kernel size
        self.assertEqual(model.layers[1].kernel_size, (3, 3))

        # Check that the Conv2D layer has the correct activation function
        self.assertEqual(model.layers[1].activation, activations.relu)

        # Create a new model with all activation functions
        vgsl_spec_string = ("None,64,None,1 Cs3,3,32 Ct3,3,32 Cr3,3,32 "
                            "Cl3,3,32 Cm3,3,32 O1s10")
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Check that the Conv2D layers have the correct activation functions
        self.assertEqual(model.layers[1].activation, activations.sigmoid)
        self.assertEqual(model.layers[2].activation, activations.tanh)
        self.assertEqual(model.layers[3].activation, activations.relu)
        self.assertEqual(model.layers[4].activation, activations.linear)
        self.assertEqual(model.layers[5].activation, activations.softmax)

    def test_maxpool_layer(self):
        vgsl_spec_string = "None,64,None,1 Mp2,2,2,2 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertIsInstance(model.layers[1], layers.MaxPooling2D)

        # Calculate the correct pool size
        input_dimension = 64
        pool_size = 2
        stride = 2
        padding = 0

        # Calculate the correct pool size
        output_dimension = (input_dimension - pool_size +
                            2 * padding) // stride + 1

        # Create a dummy input to check the output shape of the MaxPooling2D
        # layer
        dummy_input = np.random.random((1, 64, 64, 1))
        avgpool_output = model.layers[1](dummy_input)

        # Check the output shape of the MaxPooling2D layer
        _, height, width, _ = avgpool_output.shape
        self.assertEqual(height, output_dimension)
        self.assertEqual(width, output_dimension)

    def test_avgpool_layer(self):
        vgsl_spec_string = "None,64,None,1 Ap2,2,2,2 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertIsInstance(model.layers[1], layers.AvgPool2D)

        # Calculate the correct pool size
        input_dimension = 64
        pool_size = 2
        stride = 2
        padding = 0

        output_dimension = (input_dimension - pool_size +
                            2 * padding) // stride + 1

        # Create a dummy input to check the output shape of the AvgPool2D layer
        dummy_input = np.random.random((1, 64, 64, 1))
        avgpool_output = model.layers[1](dummy_input)

        # Check the output shape of the AvgPool2D layer
        _, height, width, _ = avgpool_output.shape
        self.assertEqual(height, output_dimension)
        self.assertEqual(width, output_dimension)

    def test_reshape_layer(self):
        vgsl_spec_string = "None,64,None,1 Rc O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertIsInstance(model.layers[1], layers.Reshape)

        # Calculate the correct target shape
        expected_shape = (1, 64, 64)

        # Create a dummy input to check the output shape of the Reshape layer
        dummy_input = np.random.random((1, 64, 64, 1))
        reshape_output = model.layers[1](dummy_input)

        # Check the output shape of the Reshape layer
        actual_shape = reshape_output.shape
        self.assertEqual(actual_shape, expected_shape)

    def test_fully_connected_layer(self):
        vgsl_spec_string = "None,64,None,1 Fs128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Basic tests
        self.assertIsInstance(model.layers[1], layers.Dense)
        self.assertEqual(model.layers[1].units, 128)

        # Create a new model with all activation functions
        vgsl_spec_string = "None,64,None,1 Fs128 Ft128 Fr128 Fl128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Check that the Dense layers have the correct activation functions
        self.assertEqual(model.layers[1].activation, activations.sigmoid)
        self.assertEqual(model.layers[2].activation, activations.tanh)
        self.assertEqual(model.layers[3].activation, activations.relu)
        self.assertEqual(model.layers[4].activation, activations.linear)
        self.assertEqual(model.layers[5].activation, activations.softmax)

    def test_lstm_layer(self):
        vgsl_spec_string = "None,64,None,1 Rc Lfs128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.LSTM)

        # Layer-specific tests
        self.assertEqual(model.layers[2].units, 128)
        self.assertEqual(model.layers[2].go_backwards, False)
        self.assertEqual(model.layers[2].return_sequences, True)

        # Check backwards LSTM with return_sequences
        vgsl_spec_string = "None,64,None,1 Rc Lr128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].go_backwards, True)
        self.assertEqual(model.layers[2].return_sequences, False)

    def test_gru_layer(self):
        vgsl_spec_string = "None,64,None,1 Rc Gfs128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.GRU)

        # Layer-specific tests
        self.assertEqual(model.layers[2].units, 128)
        self.assertEqual(model.layers[2].go_backwards, False)
        self.assertEqual(model.layers[2].return_sequences, True)

        # Check backwards GRU with return_sequences
        vgsl_spec_string = "None,64,None,1 Rc Gr128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].go_backwards, True)
        self.assertEqual(model.layers[2].return_sequences, False)

    def test_bidirectional_layer(self):
        vgsl_spec_string = "None,64,None,1 Rc Bgs128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.Bidirectional)
        self.assertIsInstance(model.layers[2].layer, layers.GRU)
        self.assertEqual(model.layers[2].layer.units, 128)

        vgsl_spec_string = "None,64,None,1 Rc Bls128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.Bidirectional)
        self.assertIsInstance(model.layers[2].layer, layers.LSTM)
        self.assertEqual(model.layers[2].layer.units, 128)

    def test_residual_block(self):
        vgsl_spec_string = "None,64,None,1 RB3,3,16 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[1], self.ResidualBlock)

        # Layer-specific tests
        self.assertEqual(model.layers[1].conv1.filters, 16)
        self.assertEqual(model.layers[1].conv1.kernel_size, (3, 3))
        self.assertEqual(model.layers[1].conv2.filters, 16)
        self.assertEqual(model.layers[1].conv2.kernel_size, (3, 3))

        # Create a model with downsampling
        vgsl_spec_string = "None,64,None,1 RBd3,3,16 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Check that the downsampling layer exists
        self.assertIsInstance(model.layers[1].conv3, layers.Conv2D)
        self.assertEqual(model.layers[1].conv3.filters, 16)
        self.assertEqual(model.layers[1].conv3.kernel_size, (1, 1))
        self.assertEqual(model.layers[1].conv3.strides, (2, 2))

        # Check that conv1 also has strides of 2
        self.assertEqual(model.layers[1].conv1.strides, (2, 2))

    def test_dropout_layer(self):
        vgsl_spec_string = "None,64,None,1 D50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[1], layers.Dropout)
        self.assertEqual(model.layers[1].rate, 0.5)

    def test_output_layer(self):
        vgsl_spec_string = "None,64,None,1 Cr3,3,32 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[-2], layers.Dense)
        self.assertIsInstance(model.layers[-1], layers.Activation)

        # Check that the output layer has the correct number of units
        self.assertEqual(model.layers[-2].units, 10)

        # Create a new model with different activation function and units
        vgsl_spec_string = "None,64,None,1 Cr3,3,32 O1s5"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        # Check that the output layer has the correct number of units
        self.assertEqual(model.layers[-2].units, 5)

        # TODO: CTCLayer


if __name__ == "__main__":
    unittest.main()