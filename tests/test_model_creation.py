# Imports

# > Third party dependencies
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import activations

# > Standard library
import logging
import unittest
from pathlib import Path
import sys


class VGSLModelGeneratorTest(unittest.TestCase):
    """
    Tests for creating a new model.

    Test coverage:
        1. `test_create_simple_model`: Test model creation with a simple
            VGSL-spec string.
        2. `test_conv2d_layer`: Test model creation with a Conv2D layer.
        3. `test_conv2d_error_handling`: Test error handling for Conv2D layer.
        4. `test_maxpool_layer`: Test model creation with a MaxPooling2D layer.
        5. `test_maxpool_error_handling`: Test error handling for
            MaxPooling2D layer.
        6. `test_avgpool_layer`: Test model creation with a AvgPool2D layer.
        7. `test_avgpool_error_handling`: Test error handling for
            AvgPool2D layer.
        8. `test_reshape_layer`: Test model creation with a Reshape layer.
        9. `test_reshape_error_handling`: Test error handling for Reshape
            layer.
        10. `test_fully_connected_layer`: Test model creation with a Fully
            connected layer.
        11. `test_fully_connected_error_handling`: Test error handling for
            Fully connected layer.
        12. `test_lstm_layer`: Test model creation with a LSTM layer.
        13. `test_lstm_error_handling`: Test error handling for LSTM layer.
        14. `test_gru_layer`: Test model creation with a GRU layer.
        15. `test_gru_error_handling`: Test error handling for GRU layer.
        16. `test_bidirectional_layer`: Test model creation with a
            Bidirectional layer.
        17. `test_bidirectional_error_handling`: Test error handling for
            Bidirectional layer.
        18. `test_residual_block`: Test model creation with a Residual block.
        19. `test_residual_block_error_handling`: Test error handling for
            Residual block.
        20. `test_dropout_layer`: Test model creation with a Dropout layer.
        21. `test_dropout_error_handling`: Test error handling for Dropout
            layer.
        22. `test_output_layer`: Test model creation with an Output layer.
        23. `test_output_error_handling`: Test error handling for Output layer.
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
        self.assertEqual(model.layers[1].activation, activations.softmax)
        self.assertEqual(model.layers[2].activation, activations.tanh)
        self.assertEqual(model.layers[3].activation, activations.relu)
        self.assertEqual(model.layers[4].activation, activations.linear)
        self.assertEqual(model.layers[5].activation, activations.sigmoid)

    def test_conv2d_error_handling(self):
        # Check that an error is raised when an invalid number of parameters
        # is specified
        vgsl_spec_string = "None,64,None,1 C3,32 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertIn("Conv layer C3,32 has too few parameters.",
                      str(context.exception))

        vgsl_spec_string = "None,64,None,1 C3,3,2,2,32,4 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertIn(
            "Conv layer C3,3,2,2,32,4 has too many parameters.",
            str(context.exception))

        # Check that an error is raised when an invalid activation function is
        # specified
        vgsl_spec_string = "None,64,None,1 Cz3,3,32 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual("Invalid activation function specified in Cz3,3,32",
                         str(context.exception))

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

    def test_maxpool_error_handling(self):
        # Check that an error is raised when an invalid number of parameters
        # is specified
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 Mp2,2,2 O1s10"
            self.VGSLModelGenerator(vgsl_spec_string)

        self.assertEqual(str(context.exception),
                         "MaxPooling layer Mp2,2,2 does not have the expected "
                         "number of parameters. Expected format: Mp<pool_x>,"
                         "<pool_y>,<stride_x>,<stride_y>")

        # Check that an error is raised when an invalid value is specified
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 Mp-2,2,2,2 O1s10"
            self.VGSLModelGenerator(vgsl_spec_string)

        self.assertEqual(str(context.exception),
                         "Invalid values for pooling or stride in Mp-2,2,2,2. "
                         "All values should be positive integers.")

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

    def test_avgpool_error_handling(self):
        # Check that an error is raised when an invalid number of parameters
        # is specified
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 Ap2,2,2 O1s10"
            self.VGSLModelGenerator(vgsl_spec_string)

        self.assertEqual(str(context.exception),
                         "AvgPool layer Ap2,2,2 does not have the expected "
                         "number of parameters. Expected format: Ap<pool_x>,"
                         "<pool_y>,<stride_x>,<stride_y>")

        # Check that an error is raised when an invalid value is specified
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 Ap-2,2,2,2 O1s10"
            self.VGSLModelGenerator(vgsl_spec_string)

        self.assertEqual(str(context.exception),
                         "Invalid values for pooling or stride in Ap-2,2,2,2. "
                         "All values should be positive integers.")

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

    def test_reshape_error_handling(self):
        # Test unexpected format
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 R O1s10"
            model_generator = self.VGSLModelGenerator(vgsl_spec_string)
            model_generator.build()
        self.assertEqual(str(context.exception),
                         "Reshape layer R is of unexpected format. Expected "
                         "format: Rc.")

        # Test incorrectly specified reshape layer
        with self.assertRaises(ValueError) as context:
            vgsl_spec_string = "None,64,None,1 Rx O1s10"
            model_generator = self.VGSLModelGenerator(vgsl_spec_string)
            model_generator.build()
        self.assertEqual(str(context.exception),
                         "Reshape operation Rx is not supported.")

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
        self.assertEqual(model.layers[1].activation, activations.softmax)
        self.assertEqual(model.layers[2].activation, activations.tanh)
        self.assertEqual(model.layers[3].activation, activations.relu)
        self.assertEqual(model.layers[4].activation, activations.linear)
        self.assertEqual(model.layers[5].activation, activations.softmax)

    def test_fully_connected_error_handling(self):
        # Test for malformed VGSL specification string for the dense layer
        # No activation
        vgsl_spec_string = "None,64,None,1 F128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dense layer F128 is of unexpected format. Expected "
                         "format: F(s|t|r|l|m)<d>.")
        # No neurons
        vgsl_spec_string = "None,64,None,1 Fs O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dense layer Fs is of unexpected format. Expected "
                         "format: F(s|t|r|l|m)<d>.")

        # Test for invalid activation function
        vgsl_spec_string = "None,64,None,1 Fz128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid activation 'z' for Dense layer Fz128.")

        # Test for invalid number of neurons (<= 0)
        vgsl_spec_string = "None,64,None,1 Fs-100 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid number of neurons -100 for Dense layer "
                         "Fs-100.")

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

        # Check dropout 0.5 LSTM with recurrent_dropout of 0
        vgsl_spec_string = "None,64,None,1 Rc Lr128,D50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0.50)
        self.assertEqual(model.layers[2].recurrent_dropout, 0)

        # Check recurrent dropout 0.5 LSTM with dropout of 0
        vgsl_spec_string = "None,64,None,1 Rc Lr128,Rd50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0)
        self.assertEqual(model.layers[2].recurrent_dropout, 0.50)

        # Check recurrent dropout 0.5 LSTM and dropout of 0.5
        vgsl_spec_string = "None,64,None,1 Rc Lr128,D42,Rd34 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0.42)
        self.assertEqual(model.layers[2].recurrent_dropout, 0.34)

    def test_lstm_error_handling(self):
        # Missing direction
        vgsl_spec_string = "None,64,None,1 L128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "LSTM layer L128 is of unexpected format. Expected "
                         "format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid direction
        vgsl_spec_string = "None,64,None,1 Lx128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "LSTM layer Lx128 is of unexpected format. Expected "
                         "format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Missing number of units
        vgsl_spec_string = "None,64,None,1 Lf O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "LSTM layer Lf is of unexpected format. Expected "
                         "format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid number of units (negative)
        vgsl_spec_string = "None,64,None,1 Lf-128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid number of units -128 for LSTM layer Lf-128.")

        # Invalid dropout
        vgsl_spec_string = "None,64,None,1 Lf128,D 20 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "LSTM layer Lf128,D is of unexpected format. "
                         "Expected format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 Lf128,D101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid recurrent dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 Lf128,Rd101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Recurrent dropout rate must be in the range "
                         "[0, 100].")

        # Invalid order of dropouts ("regular dropout" should come before
        # recurrent dropout)
        vgsl_spec_string = "None,64,None,1 Lf128,Rd24,D40 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "LSTM layer Lf128,Rd24,D40 is of unexpected format. "
                         "Expected format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

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

        # Check dropout 0.5 Gru with recurrent_dropout of 0
        vgsl_spec_string = "None,64,None,1 Rc Gr128,D50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0.50)
        self.assertEqual(model.layers[2].recurrent_dropout, 0)

        # Check recurrent dropout Gru 0.5 with dropout of 0
        vgsl_spec_string = "None,64,None,1 Rc Gr128,Rd50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0)
        self.assertEqual(model.layers[2].recurrent_dropout, 0.50)

        # Check recurrent dropout Gru 0.5 and dropout of 0.5
        vgsl_spec_string = "None,64,None,1 Rc Gr128,D42,Rd34 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].dropout, 0.42)
        self.assertEqual(model.layers[2].recurrent_dropout, 0.34)

    def test_gru_error_handling(self):
        # Missing direction
        vgsl_spec_string = "None,64,None,1 G128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "GRU layer G128 is of unexpected format. Expected "
                         "format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid direction
        vgsl_spec_string = "None,64,None,1 Gx128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "GRU layer Gx128 is of unexpected format. Expected "
                         "format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid number of units (negative)
        vgsl_spec_string = "None,64,None,1 Gf-128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid number of units -128 for GRU layer Gf-128.")

        # Invalid dropout
        vgsl_spec_string = "None,64,None,1 Gf128,D 20 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "GRU layer Gf128,D is of unexpected format. Expected "
                         "format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        # Invalid dropout rate (negative value)
        vgsl_spec_string = "None,64,None,1 Gf128,D-50 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 Gf128,D101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid recurrent dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 Gf128,Rd101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Recurrent dropout rate must be in the range "
                         "[0, 100].")

        # Invalid order of dropouts ("regular dropout" should come before
        # recurrent dropout)
        vgsl_spec_string = "None,64,None,1 Gf128,Rd24,D40 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "GRU layer Gf128,Rd24,D40 is of unexpected format. "
                         "Expected format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

    def test_bidirectional_layer(self):
        vgsl_spec_string = "None,64,None,1 Rc Bg128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.Bidirectional)
        self.assertIsInstance(model.layers[2].layer, layers.GRU)
        self.assertEqual(model.layers[2].layer.units, 128)

        vgsl_spec_string = "None,64,None,1 Rc Bl128 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[2], layers.Bidirectional)
        self.assertIsInstance(model.layers[2].layer, layers.LSTM)
        self.assertEqual(model.layers[2].layer.units, 128)

        vgsl_spec_string = "None,64,None,1 Rc Bl128,D50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].layer.dropout, 0.50)
        self.assertEqual(model.layers[2].layer.recurrent_dropout, 0)

        vgsl_spec_string = "None,64,None,1 Rc Bl128,Rd50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].layer.dropout, 0)
        self.assertEqual(model.layers[2].layer.recurrent_dropout, 0.50)

        vgsl_spec_string = "None,64,None,1 Rc Bl128,D42,Rd34 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()

        self.assertEqual(model.layers[2].layer.dropout, 0.42)
        self.assertEqual(model.layers[2].layer.recurrent_dropout, 0.34)

    def test_bidirectional_error_handling(self):
        # Invalid format
        vgsl_spec_string = "None,64,None,1 Rc B128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer B128 is of unexpected format. "
                         "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] where "
                         "'g' stands for GRU, 'l' stands for LSTM, 'n' is the "
                         "number of units, 'rate' is the (recurrent) dropout "
                         "rate.")

        # Invalid RNN layer type
        vgsl_spec_string = "None,64,None,1 Rc Bx128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer Bx128 is of unexpected format. "
                         "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] where "
                         "'g' stands for GRU, 'l' stands for LSTM, 'n' is the "
                         "number of units, 'rate' is the (recurrent) dropout "
                         "rate.")

        # Invalid number of units (negative)
        vgsl_spec_string = "None,64,None,1 Rc Bg-128 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid number of units -128 for layer Bg-128.")

        # Invalid dropout
        vgsl_spec_string = "None,64,None,1 Bl128,D 20 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer Bl128,D is of unexpected format. "
                         "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] where "
                         "'g' stands for GRU, 'l' stands for LSTM, 'n' is the "
                         "number of units, 'rate' is the (recurrent) dropout "
                         "rate.")

        # Invalid dropout rate (negative value)
        vgsl_spec_string = "None,64,None,1 Bl128,D-50 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 Bl128,D101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid order of dropouts ("regular dropout" should come before
        # recurrent dropout)
        vgsl_spec_string = "None,64,None,1 Bl128,Rd24,D40 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer Bl128,Rd24,D40 is of unexpected format. "
                         "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] where "
                         "'g' stands for GRU, 'l' stands for LSTM, 'n' is the "
                         "number of units, 'rate' is the (recurrent) dropout "
                         "rate.")

    # def test_residual_block(self):
    #     vgsl_spec_string = "None,64,None,1 RB3,3,16 O1s10"
    #     model_generator = self.VGSLModelGenerator(vgsl_spec_string)
    #     model = model_generator.build()
    #     self.assertIsInstance(model.layers[1], self.ResidualBlock)
    # 
    #     # Layer-specific tests
    #     self.assertEqual(model.layers[1].conv1.filters, 16)
    #     self.assertEqual(model.layers[1].conv1.kernel_size, (3, 3))
    #     self.assertEqual(model.layers[1].conv2.filters, 16)
    #     self.assertEqual(model.layers[1].conv2.kernel_size, (3, 3))
    # 
    #     # Create a model with downsampling
    #     vgsl_spec_string = "None,64,None,1 RBd3,3,16 O1s10"
    #     model_generator = self.VGSLModelGenerator(vgsl_spec_string)
    #     model = model_generator.build()
    # 
    #     # Check that the downsampling layer exists
    #     self.assertIsInstance(model.layers[1].conv3, layers.Conv2D)
    #     self.assertEqual(model.layers[1].conv3.filters, 16)
    #     self.assertEqual(model.layers[1].conv3.kernel_size, (1, 1))
    #     self.assertEqual(model.layers[1].conv3.strides, (2, 2))
    # 
    #     # Check that conv1 also has strides of 2
    #     self.assertEqual(model.layers[1].conv1.strides, (2, 2))

    def test_residual_block_error_handling(self):
        # Invalid format
        vgsl_spec_string = "None,64,None,1 RBd O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer RBd is of unexpected format. Expected format: "
                         "RB[d]<x>,<y>,<d>.")

        # Invalid parameters (negative values)
        vgsl_spec_string = "None,64,None,1 RB-3,3,-16 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Invalid parameters x=-3, y=3, d=-16 in layer "
                         "RB-3,3,-16. All values should be positive integers.")

        # Missing parameters
        vgsl_spec_string = "None,64,None,1 RB3,3 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer RB3,3 is of unexpected format. Expected "
                         "format: RB[d]<x>,<y>,<d>.")

    def test_dropout_layer(self):
        vgsl_spec_string = "None,64,None,1 D50 O1s10"
        model_generator = self.VGSLModelGenerator(vgsl_spec_string)
        model = model_generator.build()
        self.assertIsInstance(model.layers[1], layers.Dropout)
        self.assertEqual(model.layers[1].rate, 0.5)

    def test_dropout_error_handling(self):
        # Invalid format
        vgsl_spec_string = "None,64,None,1 D O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer D is of unexpected format. Expected format: "
                         "D<rate> where rate is between 0 and 100.")

        # Invalid dropout rate (negative value)
        vgsl_spec_string = "None,64,None,1 D-50 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

        # Invalid dropout rate (value greater than 100)
        vgsl_spec_string = "None,64,None,1 D101 O1s10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Dropout rate must be in the range [0, 100].")

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

    def test_output_error_handling(self):
        # Invalid format
        vgsl_spec_string = "None,64,None,1 OXs10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Layer OXs10 is of unexpected format. Expected "
                         "format: O[210](l|s)<n>.")

        # Invalid linearity
        vgsl_spec_string = "None,64,None,1 O1x10"
        with self.assertRaises(ValueError) as context:
            self.VGSLModelGenerator(vgsl_spec_string)
        self.assertEqual(str(context.exception),
                         "Output layer linearity x is not supported.")


if __name__ == "__main__":
    unittest.main()
