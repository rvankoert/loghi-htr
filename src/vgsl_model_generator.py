# Imports

# > Standard library
import argparse
import re
import logging

# > Local dependencies
from custom_layers import CTCLayer, ResidualBlock

# > Third party dependencies
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, Input


class VGSLModelGenerator:
    """
    Generates a VGSL (Variable-size Graph Specification Language) model based
    on a given specification string.

    VGSL is a domain-specific language that allows the rapid specification of
    neural network architectures. This class provides a way to generate and
    initialize a model using either a predefined model from the library or a
    custom VGSL specification string. It supports various layers like Conv2D,
    LSTM, GRU, Dense, etc.

    Parameters
    ----------
    model : str
        VGSL spec string defining the model architecture. If the string starts
        with "model", it attempts to pull the model from the predefined
        library.
    name : str, optional
        Name of the model. Defaults to the given `model` string or
        "custom_model" if it's a VGSL spec string.
    channels : int, optional
        Number of input channels. Overrides the channels specified in the
        VGSL spec string if provided.
    output_classes : int, optional
        Number of output classes. Overrides the number of classes specified in
        the VGSL spec string if provided.

    Attributes
    ----------
    model_library : dict
        Dictionary of predefined models with their VGSL spec strings.
    model_name : str
        Name of the model.
    history : list
        A list that keeps track of the order of layers added to the model.
    selected_model_vgsl_spec : list
        List of individual layers/components from the VGSL spec string.
    inputs : tf.layers.Input
        Input layer of the model.

    Raises
    ------
    KeyError
        If the provided model string is not found in the predefined model
        library.
    ValueError
        If there's an issue with the VGSL spec string format or unsupported
        operations.

    Examples
    --------
    >>> vgsl_gn = VGSLModelGenerator("None,64,None,1 Cr3,3,32 Mp2,2,2,2 O1s92")
    >>> model = vgsl_gn.build()
    >>> model.summary()
    """

    def __init__(self,
                 model: str,
                 name: str = None,
                 channels: int = None,
                 output_classes: int = None):
        """
        Initialize the VGSLModelGenerator instance.

        Parameters
        ----------
        model : str
            VGSL spec string or model name from the predefined library.
        name : str, optional
            Custom name for the model. If not provided, uses the model name
            or "custom_model" for VGSL specs.
        channels : int, optional
            Number of input channels. If provided, overrides the channels
            from the VGSL spec.
        output_classes : int, optional
            Number of output classes. If provided, overrides the number from
            the VGSL spec.

        Raises
        ------
        KeyError:
            If the model name is not found in the predefined library.
        ValueError:
            If there's an issue with the VGSL spec string format or unsupported
            operations.
        """

        self._initializer = initializers.GlorotNormal(seed=42)
        self._channel_axis = -1
        self.model_library = VGSLModelGenerator.get_model_libary()

        if model is None:
            raise ValueError("No model provided. Please provide a model name "
                             "from the model library or a VGSL-spec string.")

        if model.startswith("model"):
            try:
                logging.info("Pulling model from model library")
                model_string = self.model_library[model]
                self.init_model_from_string(model_string,
                                            channels,
                                            output_classes)
                self.model_name = name if name else model

            except KeyError:
                raise KeyError("Model not found in model library")
        else:
            try:
                logging.info("Found VGSL-Spec String, testing validity...")
                self.init_model_from_string(model,
                                            channels,
                                            output_classes)
                self.model_name = name if name else "custom_model"

            except (TypeError, AttributeError) as e:
                raise ("Something is wrong with the input string, "
                       "please check the VGSL-spec formatting "
                       "with the documentation.") from e

    def init_model_from_string(self,
                               vgsl_spec_string: str,
                               channels: int = None,
                               output_classes: int = None) -> None:
        """
        Initialize the model based on the given VGSL spec string. This method
        parses the string and creates the model layer by layer.

        Parameters
        ----------
        vgsl_spec_string : str
            VGSL spec string defining the model architecture.
        channels : int, optional
            Number of input channels. Overrides the channels specified in the
            VGSL spec string if provided.
        output_classes : int, optional
            Number of output classes. Overrides the number of classes specified
            in the VGSL spec string if provided.

        Raises
        ------
        ValueError:
            If there's an issue with the VGSL spec string format or unsupported
            operations.
        """

        logging.info("Initializing model")
        self.history = []
        self.selected_model_vgsl_spec = vgsl_spec_string.split()

        # Check if the first layer is an input layer
        pattern = r'^(None|\d+),(None|\d+),(None|\d+),(None|\d+)$'
        if re.match(pattern, self.selected_model_vgsl_spec[0]):
            self.inputs = self.make_input_layer(
                self.selected_model_vgsl_spec[0], channels)
            starting_index = 1
        else:
            self.inputs = None
            starting_index = 0

        for index, layer in \
                enumerate(self.selected_model_vgsl_spec[starting_index:]):
            logging.debug(layer)
            if layer.startswith('C'):
                setattr(self, f"conv2d_{index}", self.conv2d_generator(layer))
                self.history.append(f"conv2d_{index}")
            elif layer.startswith('Bn'):
                setattr(self, f"batchnorm_{index}", layers.BatchNormalization(
                    axis=self._channel_axis))
                self.history.append(f"batchnorm_{index}")
            elif layer.startswith('L'):
                setattr(self, f"lstm_{index}", self.lstm_generator(layer))
                self.history.append(f"lstm_{index}")
            elif layer.startswith('F'):
                setattr(self, f"dense{index}", self.fc_generator(layer))
                self.history.append(f"dense{index}")
            elif layer.startswith('B'):
                setattr(self, f"bidirectional_{index}",
                        self.bidirectional_generator(layer))
                self.history.append(f"bidirectional_{index}")
            elif layer.startswith('G'):
                setattr(self, f"gru_{index}", self.gru_generator(layer))
                self.history.append(f"gru_{index}")
            elif layer.startswith('Mp'):
                setattr(self, f"maxpool_{index}",
                        self.maxpool_generator(layer))
                self.history.append(f"maxpool_{index}")
            elif layer.startswith('Ap'):
                setattr(self, f"avgpool_{index}",
                        self.avgpool_generator(layer))
                self.history.append(f"avgpool_{index}")
            elif layer.startswith('RB'):
                # setattr(self, f"ResidualBlock_{index}",
                self.residual_block_generator(layer, index)
                # self.history.append(f"ResidualBlock_{index}")
            elif layer.startswith('D'):
                setattr(self, f"dropout_{index}",
                        self.dropout_generator(layer))
                self.history.append(f"dropout_{index}")
            elif layer.startswith('R'):
                self.history.append(f"reshape_{index}_{layer}")
            elif layer.startswith('O'):
                setattr(self, f"output_{index}",
                        self.get_output_layer(layer, output_classes))
                self.history.append(f"output_{index}")
            else:
                raise ValueError(f"The current layer: {layer} is not "
                                 "recognised, please check for correct "
                                 "formatting in the VGSL-Spec")

    def build(self) -> tf.keras.models.Model:
        """
        Build the model based on the VGSL spec string.

        Returns
        -------
        tf.keras.models.Model
            The built model.
        """

        logging.info("Building model for: %s", self.selected_model_vgsl_spec)
        if self.inputs is None:
            raise ValueError("No input layer found. Please check the "
                             "VGSL-spec string.")

        x = self.inputs
        residual_block_x = None
        for index, layer in enumerate(self.history):
            if layer.startswith("reshape"):
                x = self.reshape_generator(layer.split("_")[2], x)(x)
            else:
                if layer.startswith('conv') and layer.endswith("_conv1_rb"):
                    residual_block_x = x
                if not layer.lower().startswith('elu_'):
                    x = getattr(self, layer)(x)
                if layer.lower().startswith('conv') and getattr(self, layer).name.endswith("_conv3_rb"):
                    residual_block_x = x
                if layer.startswith('elu') \
                        and getattr(self, layer).name.startswith("elu_") \
                        and getattr(self, layer).name.endswith("_rb"):
                    print('layer '+layer)            #elu
                    x = layers.Add()([residual_block_x, x])
                    residual_block_x = None
        output = layers.Activation('linear', dtype=tf.float32)(x)

        logging.info("Model has been built\n")

        return models.Model(inputs=self.inputs,
                            outputs=output,
                            name=self.model_name)

    #######################
    #    Model Library    #
    #######################

    @ staticmethod
    def get_model_libary() -> dict:
        """
        Returns a dictionary of predefined models with their VGSL spec strings.

        Returns
        -------
        dict
            Dictionary of predefined models with their VGSL spec strings.
        """

        model_library = {
            "modelkeras":
                ("None,64,None,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc "
                 "Fl64 D20 Bl128 D20 Bl64 D20 O1s92"),
            "model9":
                ("None,64,None,1 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
                 "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Bl256,D50 Bl256,D50 "
                 "Bl256,D50 Bl256,D50 Bl256,D50 O1s92"),
            "model10":
                ("None,64,None,4 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
                 "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Bl256,D50 Bl256,D50 "
                 "Bl256,D50 Bl256,D50 Bl256,D50 O1s92"),
            "model11":
                ("None,64,None,1 Cr3,3,24 Bn Ap2,2,2,2 Cr3,3,48 Bn Cr3,3,96 Bn"
                 "Ap2,2,2,2 Cr3,3,96 Bn Ap2,2,2,2 Rc Bl256 Bl256 Bl256 "
                 "Bl256 Bl256 Fe1024 O1s92"),
            "model12":
                ("None,64,None,1 Cr1,3,12 Bn Cr3,3,48 Bn Mp2,2,2,2 Cr3,3,96 "
                 "Cr3,3,96 Bn Mp2,2,2,2 Rc Bl256 Bl256 Bl256 Bl256 Bl256 "
                 "O1s92"),
            "model13":
                ("None,64,None,1 Cr1,3,12 Bn Cr3,1,24 Bn Mp2,2,2,2 Cr1,3,36 "
                 "Bn Cr3,1,48 Bn Cr1,3,64 Bn Cr3,1,96 Bn Cr1,3,96 Bn Cr3,1,96 "
                 "Bn Rc Bl256 Bl256 Bl256 Bl256 Bl256 O1s92"),
            "model14":
                ("None,64,None,1 Ce3,3,24 Bn Mp2,2,2,2 Ce3,3,36 Bn Mp2,2,2,2 "
                 "Ce3,3,64 Bn Mp2,2,2,2 Ce3,3,96 Bn Ce3,3,128 Bn Rc Bl256,D50 "
                 "Bl256,D50 Bl256,D50 Bl256,D50 Bl256,D50 O1s92"),
            "model15":
                ("None,64,None,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
                 "Ce3,3,32 Bn Ce3,3,48 Bn Rc Bg256,D50 Bg256,D50 Bg256,D50 "
                 "Bg256,D50 Bg256,D50 O1s92"),
            "model16":
                ("None,64,None,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
                 "Ce3,3,32 Bn Ce3,3,48 Bn Rc Lfs128,D50 Lfs128,D50 Lfs128,D50 "
                 "Lfs128,D50 Lfs128,D50 O1s92"),
            # "model17":
            #     ("None,64,None,4 Bn Ce3,3,16 RB3,3,16 RB3,3,16 RBd3,3,32 "
            #      "RB3,3,32 RB3,3,32 RB3,3,32 RB3,3,32 RBd3,3,64 RB3,3,64 "
            #      "RB3,3,64 RB3,3,64 RB3,3,64 RBd3,3,128 RB3,3,128 Rc "
            #      "Bl256,D50 Bl256,D50 Bl256,D50 Bl256,D50 Bl256,D50 "
            #      "O1s92")
            "model17":
                ("None,64,None,4 Bn Ce3,3,16 "
                 "RB3,3,16 RB3,3,16 "
                 "RBd3,3,32 RB3,3,32 RB3,3,32 RB3,3,32 RB3,3,32"
                 "RBd3,3,64 RB3,3,64 RB3,3,64 RB3,3,64 RB3,3,64 "
                 "RBd3,3,128 RB3,3,128 "
                 "Rc "
                 "Bl256,D20 Bl256,D20 Bl256,D20 "
                 "O1s92")
        }

        return model_library

    @ staticmethod
    def print_models():
        """
        Prints all the predefined models in the model library.

        Returns
        -------
        None
        """

        model_library = VGSLModelGenerator.get_model_libary()
        print("Listing models from model library...\n")
        print("=========================================================")
        print(f"Models found: {len(model_library)}")
        print("=========================================================\n")

        for key, value in model_library.items():
            print(f"{key}\n"
                  f"{'-' * len(key)}\n"
                  f"{value}\n")

        print("=========================================================")

    ########################
    #   Helper functions   #
    ########################

    @staticmethod
    def model_to_vgsl(model: tf.keras.models.Model) -> str:
        """
        Convert a Keras model to a VGSL spec string.

        Parameters
        ----------
        model : tf.keras.models.Model
            Keras model to be converted.

        Returns
        -------
        str
            VGSL spec string.

        Raises
        ------
        ValueError
            If the model contains unsupported layers.
        """

        def get_dropout(dropout: float, recurrent_dropout: int = 0) -> str:
            """Helper function to generate dropout specifications."""

            dropout_spec = f",D{int(dropout*100)}" if dropout > 0 else ""
            recurrent_dropout_spec = f",Rd{int(recurrent_dropout*100)}" \
                if recurrent_dropout > 0 else ""

            return f"{dropout_spec}{recurrent_dropout_spec}"

        def get_stride_spec(strides: tuple) -> str:
            """Helper function to generate stride specifications."""

            return f",{strides[0]},{strides[1]}" if strides != (1, 1) else ""

        vgsl_parts = []
        activation_map = {'softmax': 's', 'tanh': 't', 'relu': 'r',
                          'elu': 'e', 'linear': 'l', 'sigmoid': 'm'}

        # Map Input layer
        # If the first layer is an InputLayer, get the input shape from the
        # second layer
        # This is only the case where we have a model created with the Keras
        # functional API
        if isinstance(model.layers[0], tf.keras.layers.InputLayer):
            input_shape = model.layers[0].input_shape[0]
            start_idx = 1
        else:
            input_shape = model.layers[0].input_shape
            start_idx = 0

        if not (len(input_shape) == 4 and
                all(isinstance(dim, (int, type(None)))
                    for dim in input_shape)):
            raise ValueError(f"Invalid input shape {input_shape}. Input shape "
                             "must be of the form (None, height, width, "
                             "channels).")

        vgsl_parts.append(
            f"{input_shape[0]},{input_shape[2]},{input_shape[1]},"
            f"{input_shape[3]}")

        # Loop through and map the rest of the layers
        for idx in range(start_idx, len(model.layers)):
            layer = model.layers[idx]
            if isinstance(layer, tf.keras.layers.Conv2D):
                act = activation_map[layer.get_config()["activation"]]
                if act is None:
                    raise ValueError(
                        "Unsupported activation function "
                        f"{layer.get_config()['activation']} in layer "
                        f"{type(layer).__name__} at position {idx}.")

                vgsl_parts.append(
                    f"C{act}{layer.kernel_size[0]},{layer.kernel_size[1]},"
                    f"{layer.filters}{get_stride_spec(layer.strides)}")

            elif isinstance(layer, tf.keras.layers.Dense):
                act = activation_map[layer.get_config()["activation"]]
                if act is None:
                    raise ValueError(
                        "Unsupported activation function "
                        f"{layer.get_config()['activation']} in layer "
                        f"{type(layer).__name__} at position {idx}.")
                prefix = "O1" if idx == len(model.layers) - 1 or isinstance(
                    model.layers[idx + 1], tf.keras.layers.Activation) else "F"

                vgsl_parts.append(f"{prefix}{act}{layer.units}")

            elif isinstance(layer, (tf.keras.layers.LSTM,
                                    tf.keras.layers.GRU)):
                act = 'L' if isinstance(layer, tf.keras.layers.LSTM) else 'G'
                direction = 'r' if layer.go_backwards else 'f'
                return_sequences = "s" if layer.return_sequences else ""

                vgsl_parts.append(
                    f"{act}{direction}{return_sequences}{layer.units}"
                    f"{get_dropout(layer.dropout, layer.recurrent_dropout)}")

            elif isinstance(layer, layers.Bidirectional):
                wrapped_layer = layer.layer
                cell_type = 'l' if isinstance(
                    wrapped_layer, tf.keras.layers.LSTM) else 'g'
                dropout = get_dropout(wrapped_layer.dropout,
                                      wrapped_layer.recurrent_dropout)

                vgsl_parts.append(
                    f"B{cell_type}{wrapped_layer.units}{dropout}")

            elif isinstance(layer, layers.BatchNormalization):
                vgsl_parts.append("Bn")

            elif isinstance(layer, layers.MaxPooling2D):
                vgsl_parts.append(
                    f"Mp{layer.pool_size[0]},{layer.pool_size[1]},"
                    f"{layer.strides[0]},{layer.strides[1]}")

            elif isinstance(layer, layers.AveragePooling2D):
                vgsl_parts.append(
                    f"Ap{layer.pool_size[0]},{layer.pool_size[1]},"
                    f"{layer.strides[0]},{layer.strides[1]}")

            elif isinstance(layer, layers.Dropout):
                vgsl_parts.append(f"D{int(layer.rate*100)}")

            elif isinstance(layer, layers.Reshape):
                vgsl_parts.append("Rc")

            elif isinstance(layer, ResidualBlock):
                downsample_spec = "d" if layer.downsample else ""

                vgsl_parts.append(
                    f"RB{downsample_spec}{layer.conv1.kernel_size[0]},"
                    f"{layer.conv1.kernel_size[1]},{layer.conv1.filters}")

            elif isinstance(layer, layers.Activation):
                # Activation layers are not included in the VGSL spec
                # but is handled in the output layer
                continue

            else:
                # If an unsupported layer type is encountered
                raise ValueError(
                    f"Unsupported layer type {type(layer).__name__} at "
                    f"position {idx}.")

        return " ".join(vgsl_parts)

    @ staticmethod
    def get_units_or_outputs(layer: str) -> int:
        """
        Retrieve the number of units or outputs from a layer string

        Parameters
        ----------
        layer : str
            Layer string from the VGSL spec.

        Returns
        -------
        int
            Number of units or outputs.
        """

        match = re.findall(r'\d+', layer)
        if not match:
            raise ValueError(
                f"No units or outputs found in layer string {layer}.")
        return int(match[-1])

    @ staticmethod
    def get_activation_function(nonlinearity: str) -> str:
        """
        Retrieve the activation function from the layer string

        Parameters
        ----------
        nonlinearity : str
            Non-linearity string from the layer string.

        Returns
        -------
        str
            Activation function.
        """

        mapping = {'s': 'softmax', 't': 'tanh', 'r': 'relu',
                   'e': 'elu', 'l': 'linear', 'm': 'sigmoid'}

        if nonlinearity not in mapping:
            raise ValueError(
                f"Unsupported nonlinearity '{nonlinearity}' provided.")

        return mapping[nonlinearity]

    def make_input_layer(self,
                         inputs: str,
                         channels: int = None) -> tf.keras.layers.Input:
        """
        Create the input layer based on the input string

        Parameters
        ----------
        inputs : str
            Input string from the VGSL spec.
        channels : int, optional
            Number of input channels.

        Returns
        -------
        tf.keras.layers.Input
            Input layer.
        """

        try:
            batch, height, width, depth = map(
                lambda x: None if x == "None" else int(x), inputs.split(","))
        except ValueError:
            raise ValueError(
                f"Invalid input string format {inputs}. Expected format: "
                "batch,height,width,depth.")

        if channels and depth != channels:
            logging.warning("Overwriting channels from input string. "
                            "Was: %s, now: %s", depth, channels)
            depth = channels
            self.selected_model_vgsl_spec[0] = (f"{batch},{height},"
                                                f"{width},{depth}")

        logging.info("Creating input layer with shape: (%s, %s, %s, %s)",
                     batch, height, width, depth)
        return Input(shape=(width, height, depth), name='image')

    #######################
    #   Layer functions   #
    #######################

    def conv2d_generator(self,
                         layer: str, name=None) -> tf.keras.layers.Conv2D:
        """
        Generate a 2D convolutional layer based on a VGSL specification string.

        The method creates a Conv2D layer based on the provided VGSL spec
        string. The string can optionally include strides, and if not provided,
        default stride values are used.

        Parameters
        ----------
        layer : str
            VGSL specification for the convolutional layer. Expected format:
            `C(s|t|r|l|m)<x>,<y>,[<s_x>,<s_y>,]<d>`
            - (s|t|r|l|m): Activation type.
            - <x>,<y>: Kernel size.
            - <s_x>,<s_y>: Optional strides (defaults to (1, 1) if not
              provided).
            - <d>: Number of filters (depth).

        Returns
        -------
        tf.keras.layers.Conv2D
            A Conv2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> conv_layer = vgsl_gn.conv2d_generator("Ct3,3,64")
        >>> type(conv_layer)
        <class 'keras.src.layers.convolutional.conv2d.Conv2D'>
        """

        # Extract convolutional parameters
        conv_filter_params = [int(match)
                              for match in re.findall(r'\d+', layer)]

        # Check if the layer format is as expected
        if len(conv_filter_params) < 3:
            raise ValueError(f"Conv layer {layer} has too few parameters. "
                             "Expected format: C<x>,<y>,<d> or C<x>,<y>,<s_x>"
                             ",<s_y>,<d>")
        elif len(conv_filter_params) > 5:
            raise ValueError(f"Conv layer {layer} has too many parameters. "
                             "Expected format: C<x>,<y>,<d> or C<x>,<y>,<s_x>,"
                             "<s_y>,<d>")

        # Get activation function
        try:
            activation = self.get_activation_function(layer[1])
        except ValueError:
            raise ValueError(
                f"Invalid activation function specified in {layer}")

        # Check parameter length and generate corresponding Conv2D layer
        if len(conv_filter_params) == 3:
            x, y, d = conv_filter_params
            logging.warning(
                "No stride provided, setting default stride of (1,1)")
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self._initializer,
                                 name=name)
        elif len(conv_filter_params) == 5:
            x, y, s_x, s_y, d = conv_filter_params
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(s_x, s_y),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self._initializer,
                                 name=name)
        else:
            raise ValueError(f"Invalid number of parameters in {layer}")

    def maxpool_generator(self,
                          layer: str) -> tf.keras.layers.MaxPooling2D:
        """
        Generate a MaxPooling2D layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the max pooling layer. Expected format:
            `Mp<x>,<y>,<s_x>,<s_y>`
            - <x>,<y>: Pool size.
            - <s_x>,<s_y>: Strides.

        Returns
        -------
        tf.keras.layers.MaxPooling2D
            A MaxPooling2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> maxpool_layer = vgsl_gn.maxpool_generator("Mp2,2,2,2")
        >>> type(maxpool_layer)
        <class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>
        """

        # Extract pooling and stride parameters
        pool_stride_params = [int(match)
                              for match in re.findall(r'-?\d+', layer)]

        # Check if the parameters are as expected
        if len(pool_stride_params) != 4:
            raise ValueError(f"MaxPooling layer {layer} does not have the "
                             "expected number of parameters. Expected format: "
                             "Mp<pool_x>,<pool_y>,<stride_x>,<stride_y>")

        pool_x, pool_y, stride_x, stride_y = pool_stride_params

        # Check if pool and stride values are valid
        if pool_x <= 0 or pool_y <= 0 or stride_x <= 0 or stride_y <= 0:
            raise ValueError(f"Invalid values for pooling or stride in "
                             f"{layer}. All values should be positive "
                             "integers.")

        return layers.MaxPooling2D(pool_size=(pool_x, pool_y),
                                   strides=(stride_x, stride_y),
                                   padding='same')

    def avgpool_generator(self,
                          layer: str) -> tf.keras.layers.AvgPool2D:
        """
        Generate an AvgPool2D layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the average pooling layer. Expected format:
            `Ap<x>,<y>,<s_x>,<s_y>`
            - <x>,<y>: Pool size.
            - <s_x>,<s_y>: Strides.

        Returns
        -------
        tf.keras.layers.AvgPool2D
            An AvgPool2D layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> avgpool_layer = vgsl_gn.avgpool_generator("Ap2,2,2,2")
        >>> type(avgpool_layer)
        <class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>
        """

        # Extract pooling and stride parameters
        pool_stride_params = [int(match)
                              for match in re.findall(r'-?\d+', layer)]

        # Check if the parameters are as expected
        if len(pool_stride_params) != 4:
            raise ValueError(f"AvgPool layer {layer} does not have the "
                             "expected number of parameters. Expected format: "
                             "Ap<pool_x>,<pool_y>,<stride_x>,<stride_y>")

        pool_x, pool_y, stride_x, stride_y = pool_stride_params

        # Check if pool and stride values are valid
        if pool_x <= 0 or pool_y <= 0 or stride_x <= 0 or stride_y <= 0:
            raise ValueError(f"Invalid values for pooling or stride in "
                             f"{layer}. All values should be positive "
                             "integers.")

        return layers.AvgPool2D(pool_size=(pool_x, pool_y),
                                strides=(stride_x, stride_y),
                                padding='same')

    def reshape_generator(self,
                          layer: str,
                          prev_layer: tf.keras.layers.Layer) \
            -> tf.keras.layers.Reshape:
        """
        Generate a reshape layer based on a VGSL specification string.

        The method reshapes the output of the previous layer based on the
        provided VGSL spec string.
        Currently, it supports collapsing the spatial dimensions into a single
        dimension.

        Parameters
        ----------
        layer : str
            VGSL specification for the reshape operation. Expected formats:
            - `Rc`: Collapse the spatial dimensions.
        prev_layer : tf.keras.layers.Layer
            The preceding layer that will be reshaped.

        Returns
        -------
        tf.keras.layers.Reshape
            A Reshape layer with the specified parameters if the operation is
            known, otherwise a string indicating the operation is not known.

        Raises
        ------
        ValueError:
            If the VGSL spec string does not match the expected format or if
            the reshape operation is unknown.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> prev_layer = vgsl_gn.make_input_layer("None,64,None,1")
        >>> reshape_layer = vgsl_gn.reshape_generator("Rc", prev_layer)
        >>> type(reshape_layer)
        <class 'keras.src.layers.reshaping.reshape.Reshape'>
        """

        # Check if the layer format is as expected
        if len(layer) < 2:
            raise ValueError(f"Reshape layer {layer} is of unexpected format. "
                             "Expected format: Rc.")

        if layer[1] == 'c':
            prev_layer_y, prev_layer_x = prev_layer.shape[-2:]
            return layers.Reshape((-1, prev_layer_y * prev_layer_x))
        else:
            raise ValueError(f"Reshape operation {layer} is not supported.")

    def fc_generator(self,
                     layer: str) -> tf.keras.layers.Dense:
        """
        Generate a fully connected (dense) layer based on a VGSL specification
        string.

        Parameters
        ----------
        layer : str
            VGSL specification for the fully connected layer. Expected format:
            `F(s|t|r|l|m)<d>`
            - `(s|t|r|l|m)`: Non-linearity type. One of sigmoid, tanh, relu,
            linear, or softmax.
            - `<d>`: Number of outputs.

        Returns
        -------
        tf.keras.layers.Dense
            A Dense layer with the specified parameters.

        Raises
        ------
        ValueError:
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> dense_layer = vgsl_gn.fc_generator("Fr64")
        >>> type(dense_layer)
        <class 'keras.src.layers.core.dense.Dense'>
        >>> dense_layer.activation
        <function relu at 0x7f8b1c0b1d30>

        Notes
        -----
        This method produces a fully connected layer that reduces the height
        and width of the input to 1, producing a single vector as output. The
        input height and width must be constant. For sliding-window operations
        that leave the input image size unchanged, use a 1x1 convolution
        (e.g., Cr1,1,64) instead of this method.
        """

        # Ensure the layer string format is as expected
        if not re.match(r'^F[a-z]-?\d+$', layer):
            raise ValueError(
                f"Dense layer {layer} is of unexpected format. Expected "
                "format: F(s|t|r|l|m)<d>."
            )

        # Check if the activation function is valid
        # or any other supported activations
        try:
            activation = self.get_activation_function(layer[1])
        except ValueError:
            raise ValueError(
                f"Invalid activation '{layer[1]}' for Dense layer "
                f"{layer}.")

        # Extract the number of neurons
        n = int(layer[2:])
        if n <= 0:
            raise ValueError(
                f"Invalid number of neurons {n} for Dense layer {layer}."
            )

        return layers.Dense(n,
                            activation=activation,
                            kernel_initializer=self._initializer)

    def lstm_generator(self,
                       layer: str) -> tf.keras.layers.LSTM:
        """
        Generate an LSTM layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the LSTM layer. Expected format:
            `L(f|r)[s]<n>[,D<rate>,Rd<rate>]`
            - `(f|r)`: Direction of LSTM. 'f' for forward, 'r' for reversed.
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.

        Returns
        -------
        tf.keras.layers.LSTM
            An LSTM layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> lstm_layer = vgsl_gn.lstm_generator("Lf64")
        >>> type(lstm_layer)
        <class 'keras.src.layers.rnn.lstm.LSTM'>
        """

        # Extract direction, summarization, and units
        match = re.match(r'L([fr])(s?)(-?\d+),?(D\d+)?,?(Rd\d+)?$', layer)
        if not match:
            raise ValueError(
                f"LSTM layer {layer} is of unexpected format. Expected "
                "format: L(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        direction, summarize, n, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        n = int(n)

        # Check if the number of units is valid
        if n <= 0:
            raise ValueError(
                f"Invalid number of units {n} for LSTM layer {layer}.")

        lstm_params = {
            "units": n,
            "return_sequences": 's' in layer,
            "go_backwards": direction == 'r',
            "kernel_initializer": self._initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout / 100
            if recurrent_dropout > 0 else 0,
        }

        return layers.LSTM(**lstm_params)

    def gru_generator(self,
                      layer: str) -> tf.keras.layers.GRU:
        """
        Generate a GRU layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the GRU layer. Expected format:
            `G(f|r)[s]<n>[,D<rate>,Rd<rate>]`
            - `(f|r)`: Direction of GRU. 'f' for forward, 'r' for reversed
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.


        Returns
        -------
        tf.keras.layers.GRU
            A GRU layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> gru_layer = vgsl_gn.gru_generator("Gf64")
        >>> type(gru_layer)
        <class 'keras.src.layers.rnn.gru.GRU'>
        """

        # Extract direction, summarization, and units
        match = re.match(r'G([fr])(s?)(-?\d+),?(D-?\d+)?,?(Rd-?\d+)?$', layer)
        if not match:
            raise ValueError(
                f"GRU layer {layer} is of unexpected format. Expected "
                "format: G(f|r)[s]<n>[,D<rate>,Rd<rate>].")

        direction, summarize, n, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        # Convert n to integer
        n = int(n)

        # Check if the number of units is valid
        if n <= 0:
            raise ValueError(
                f"Invalid number of units {n} for GRU layer {layer}.")

        gru_params = {
            "units": n,
            "return_sequences": bool(summarize),
            "go_backwards": direction == 'r',
            "kernel_initializer": self._initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout/100
            if recurrent_dropout > 0 else 0
        }

        return layers.GRU(**gru_params)

    def bidirectional_generator(self,
                                layer: str) -> tf.keras.layers.Bidirectional:
        """
        Generate a Bidirectional RNN layer based on a VGSL specification
        string.
        The method supports both LSTM and GRU layers for the bidirectional RNN.

        Parameters
        ----------
        layer : str
            VGSL specification for the Bidirectional layer. Expected format:
            `B(g|l)<n>[,D<rate>,Rd<rate>]`
            - `(g|l)`: Type of RNN layer. 'g' for GRU and 'l' for LSTM.
            - `<n>`: Number of units in the RNN layer.
            - `D<rate>` : Dropout rate. Should be between 0 and 100.
            - `Rd<rate>`: Recurrent dropout rate. Should be between 0 and 100.



        Returns
        -------
        tf.keras.layers.Bidirectional
            A Bidirectional RNN layer with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> bidirectional_layer = vgsl_gn.bidirectional_generator("Bl256")
        >>> type(bidirectional_layer)
        <class ''keras.src.layers.rnn.bidirectional.Bidirectional>
        >>> type(bidirectional_layer.layer)
        <class 'keras.src.layers.rnn.lstm.LSTM'>

        Notes
        -----
        The Bidirectional layer wraps an RNN layer (either LSTM or GRU) and
        runs it in both forward and backward directions.
        """

        # Extract layer type and units
        match = re.match(r'B([gl])(-?\d+),?(D-?\d+)?,?(Rd-?\d+)?$', layer)
        if not match:
            raise ValueError(f"Layer {layer} is of unexpected format. "
                             "Expected format: B(g|l)<n>[,D<rate>,Rd<rate>] "
                             "where 'g' stands for GRU, 'l' stands for LSTM, "
                             "'n' is the number of units, 'rate' is the "
                             "(recurrent) dropout rate.")

        layer_type, units, dropout, recurrent_dropout = match.groups()
        dropout = 0 if dropout is None else int(dropout.replace('D', ""))
        recurrent_dropout = 0 if recurrent_dropout is None else int(
            recurrent_dropout.replace("Rd", ""))

        # Check if the dropout is valid
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        # Check if the recurrent dropout is valid
        if recurrent_dropout < 0 or recurrent_dropout > 100:
            raise ValueError(
                "Recurrent dropout rate must be in the range [0, 100].")

        units = int(units)

        # Check if the number of units is valid
        if units <= 0:
            raise ValueError(
                f"Invalid number of units {units} for layer {layer}.")

        # Determine the RNN layer type
        rnn_layer = layers.LSTM if layer_type == 'l' else layers.GRU

        rnn_params = {
            "units": units,
            "return_sequences": True,
            "kernel_initializer": self._initializer,
            "dropout": dropout/100 if dropout > 0 else 0,
            "recurrent_dropout": recurrent_dropout/100
            if recurrent_dropout > 0 else 0
        }

        return layers.Bidirectional(rnn_layer(**rnn_params),
                                    merge_mode='concat')

    def residual_block_generator(self,
                                 layer: str, index) -> ResidualBlock:
        """
        Generate a Residual Block based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the Residual Block. Expected format:
            `RB[d]<x>,<y>,<d>`
            - `[d]`: Optional downsample flag. If provided, the block will
            downsample the input.
            - `<x>`, `<y>`: Kernel sizes in the x and y dimensions
            respectively.
            - `<d>`: Depth of the Conv2D layers within the Residual Block.

        Returns
        -------
        ResidualBlock
            A Residual Block with the specified parameters.

        Raises
        ------
        ValueError
            If the provided VGSL spec string does not match the expected
            format.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> res_block = vgsl_gn.residual_block_generator("RB3,3,64")
        >>> type(res_block)
        <class 'custom_layers.ResidualBlock'>
        """

        match = re.match(r'RB([d]?)(-?\d+),(-?\d+),(-?\d+)$', layer)
        if not match:
            raise ValueError(
                f"Layer {layer} is of unexpected format. Expected format: "
                "RB[d]<x>,<y>,<d>.")

        downsample, x, y, d = match.groups()
        x, y, d = map(int, (x, y, d))

        # Validate parameters
        if any(val <= 0 for val in [x, y, d]):
            raise ValueError(
                f"Invalid parameters x={x}, y={y}, d={d} in layer {layer}. "
                "All values should be positive integers.")

        stride = ',1,1,'
        spec = 'Ce,' + str(x)+',' + str(y) + stride + str(d)
        setattr(self, f"conv2d_{index}_conv1_rb", self.conv2d_generator(spec))
        self.history.append(f"conv2d_{index}_conv1_rb")

        spec = 'Ce,'+str(x)+','+str(y)+',1,1,' + str(d)
        setattr(self, f"conv2d_{index}_conv2_rb", self.conv2d_generator(spec))
        self.history.append(f"conv2d_{index}_conv2_rb")

        if downsample:
            spec = 'Ce,' + str(x) + ',' + str(y) + ',2,2,' + str(d)
            setattr(self, f"conv2d_{index}_conv3_rb", self.conv2d_generator(spec, name=f"conv2d_{index}_conv3_rb"))
            self.history.append(f"conv2d_{index}_conv3_rb")

        setattr(self, f"elu_{index}_rb", layers.ELU(name=f"elu_{index}_rb"))
        self.history.append(f"elu_{index}_rb")

        setattr(self, f"batchnorm_{index}_rb", layers.BatchNormalization(
            axis=self._channel_axis))
        self.history.append(f"batchnorm_{index}_rb")

    def dropout_generator(self,
                          layer: str) -> tf.keras.layers.Dropout:
        """
        Generate a Dropout layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the Dropout layer. Expected format:
            `D<rate>`
            - `<rate>`: Dropout percentage (0-100).

        Returns
        -------
        tf.keras.layers.Dropout
            A Dropout layer with the specified dropout rate.

        Raises
        ------
        ValueError
            If the layer format is unexpected or if the specified dropout rate
            is not in range [0, 100].

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> dropout_layer = vgsl_gn.dropout_generator("Do50")
        >>> type(dropout_layer)
        <class 'keras.src.layers.regularization.dropout.Dropout'>
        """

        # Validate layer format and extract dropout rate
        match = re.match(r'D(-?\d+)$', layer)
        if not match:
            raise ValueError(
                f"Layer {layer} is of unexpected format. Expected format: "
                "D<rate> where rate is between 0 and 100.")

        dropout_rate = int(match.group(1))

        # Validate dropout rate
        if dropout_rate < 0 or dropout_rate > 100:
            raise ValueError("Dropout rate must be in the range [0, 100].")

        return layers.Dropout(dropout_rate / 100)

    def get_output_layer(self,
                         layer: str,
                         output_classes: int = None) -> tf.keras.layers.Dense:
        """
        Generate an output layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the output layer. Expected format:
            `O(2|1|0)(l|s)<n>`
            - `(2|1|0)`: Dimensionality of the output.
            - `(l|s)`: Non-linearity type.
            - `<n>`: Number of output classes.
        output_classes : int
            Number of output classes to overwrite the classes defined in the
            VGSL string.

        Returns
        -------
        tf.keras.layers.Dense
            An output layer with the specified parameters.

        Raises
        ------
        ValueError
            If the output layer type specified is not supported or if an
            unsupported linearity is specified.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> output_layer = vgsl_gn.get_output_layer("O1s10", 10)
        >>> type(output_layer)
        <class 'keras.src.layers.core.dense.Dense'>
        """

        # Validate layer format
        match = re.match(r'O([210])([a-z])(\d+)$', layer)
        if not match:
            raise ValueError(
                f"Layer {layer} is of unexpected format. Expected format: "
                "O[210](l|s)<n>.")

        dimensionality, linearity, classes = match.groups()
        classes = int(classes)

        # Handle potential mismatch in specified classes and provided
        # output_classes
        if output_classes and classes != output_classes:
            logging.warning(
                "Overwriting output classes from input string. Was: %s, now: "
                "%s", classes, output_classes)
            classes = output_classes
            self.selected_model_vgsl_spec[-1] = (f"O{dimensionality}"
                                                 f"{linearity}{classes}")

        if linearity == "s":
            return layers.Dense(classes,
                                activation='softmax',
                                kernel_initializer=self._initializer)
        elif linearity == "l":
            return layers.Dense(classes,
                                activation='linear',
                                kernel_initializer=self._initializer)
        else:
            raise ValueError(
                f"Output layer linearity {linearity} is not supported.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get the VGSL spec for a given Keras model.')

    parser.add_argument('--model_dir', type=str,
                        help='Path to the directory containing the saved '
                             'model.')

    args = parser.parse_args()

    from model import CERMetric, WERMetric, CTCLoss

    # Load the model
    model = tf.keras.models.load_model(args.model_dir,
                                       custom_objects={
                                           "CERMetric": CERMetric,
                                           "WERMetric": WERMetric,
                                           "CTCLoss": CTCLoss})

    # Get the VGSL spec
    vgsl_spec = VGSLModelGenerator.model_to_vgsl(model)

    print(f"VGSL Spec: {vgsl_spec}")
