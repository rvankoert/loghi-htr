# Imports

# > Standard library
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

        super().__init__()
        self._initializer = initializers.GlorotNormal(seed=42)
        self._channel_axis = -1
        self.model_library = VGSLModelGenerator.get_model_libary()
        self.model_name = name if name else model

        if model.startswith("model"):
            try:
                logging.info("Pulling model from model library")
                model_string = self.model_library[model]
                self.init_model_from_string(model_string,
                                            channels,
                                            output_classes)
            except KeyError:
                raise KeyError("Model not found in model library")
        else:
            try:
                logging.info("Found VGSL-Spec String, testing validity...")
                self.init_model_from_string(model,
                                            channels,
                                            output_classes)

                # TODO: Add model_name argument to arg_parser.py
                self.model_name = "custom_model"

            except Exception:
                raise ValueError("Something is wrong with the input string, "
                                 "please check the VGSL-spec formatting "
                                 "with the documentation.")

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
        self.inputs = self.make_input_layer(self.selected_model_vgsl_spec[0],
                                            channels)

        for index, layer in enumerate(self.selected_model_vgsl_spec[1:]):
            logging.debug(layer)
            if layer.startswith('C'):
                setattr(self, f"conv2d{index}", self.conv2d_generator(layer))
                self.history.append(f"conv2d{index}")
            elif layer.startswith('Bn'):
                setattr(self, f"batchnorm{index}", layers.BatchNormalization(
                    axis=self._channel_axis))
                self.history.append(f"batchnorm{index}")
            elif layer.startswith('L'):
                setattr(self, f"lstm{index}", self.lstm_generator(layer))
                self.history.append(f"lstm{index}")
            elif layer.startswith('F'):
                setattr(self, f"dense{index}", self.fc_generator(layer))
                self.history.append(f"dense{index}")
            elif layer.startswith('B'):
                setattr(self, f"bidirectional{index}",
                        self.bidirectional_generator(layer))
                self.history.append(f"bidirectional{index}")
            elif layer.startswith('G'):
                setattr(self, f"gru{index}", self.gru_generator(layer))
                self.history.append(f"gru{index}")
            elif layer.startswith('Mp'):
                setattr(self, f"maxpool{index}", self.maxpool_generator(layer))
                self.history.append(f"maxpool{index}")
            elif layer.startswith('Ap'):
                setattr(self, f"avgpool{index}", self.avgpool_generator(layer))
                self.history.append(f"avgpool{index}")
            elif layer.startswith('RB'):
                setattr(self, f"ResidualBlock{index}",
                        self.residual_block_generator(layer))
                self.history.append(f"ResidualBlock{index}")
            elif layer.startswith('D'):
                setattr(self, f"dropout{index}", self.dropout_generator(layer))
                self.history.append(f"dropout{index}")
            elif layer.startswith('R'):
                self.history.append("reshape"+str(index)+"_"+(layer))
            elif layer.startswith('O1'):
                setattr(self, f"output{index}",
                        self.get_output_layer(layer, output_classes))
                self.history.append(f"output{index}")
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
        x = self.inputs
        for index, layer in enumerate(self.history):
            if (layer.startswith("reshape")):
                x = self.reshape_generator(layer.split("_")[1], x)(x)
            else:
                x = getattr(self, layer)(x)
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
                 "Fc64 D20 Lrs128 D20 Lrs64 D20 O1s92"),
            "model10":
                ("None,64,None,1 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
                 "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Grs256 Grs256 Grs256 "
                 "Grs256 Grs256 O1s92"),
            "model9":
                ("None,64,None,1 Cr3,3,24 Bn Mp2,2,2,2 Cr3,3,48 Bn Mp2,2,2,2 "
                 "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Lrs256 D20 Lrs256 D20 "
                 "Lrs256 D20 Lrs256 D20 Lrs256 D20 O1s92"),
            "model11":
                ("None,64,None,1 Cr3,3,24 Bn Ap2,2,2,2 Cr3,3,48 Bn Cr3,3,96 Bn"
                 "Ap2,2,2,2 Cr3,3,96 Bn Ap2,2,2,2 Rc Grs256 Grs256 Grs256 "
                 "Grs256 Grs256 Fe1024 O1s92"),
            "model12":
                ("None,64,None,1 Cr1,3,12 Bn Cr3,3,48 Bn Mp2,2,2,2 Cr3,3,96 "
                 "Cr3,3,96 Bn Mp2,2,2,2 Rc Grs256 Grs256 Grs256 Grs256 Grs256 "
                 "O1s92"),
            "model13":
                ("None,64,None,1 Cr1,3,12 Bn Cr3,1,24 Bn Mp2,2,2,2 Cr1,3,36 "
                 "Bn Cr3,1,48 Bn Cr1,3,64 Bn Cr3,1,96 Bn Cr1,3,96 Bn Cr3,1,96 "
                 "Bn Rc Grs256 Grs256 Grs256 Grs256 Grs256 O1s92"),
            "model14":
                ("None,64,None,1 Ce3,3,24 Bn Mp2,2,2,2 Ce3,3,36 Bn Mp2,2,2,2 "
                 "Ce3,3,64 Bn Mp2,2,2,2 Ce3,3,96 Bn Ce3,3,128 Bn Rc Grs256 "
                 "Grs256 Grs256 Grs256 Grs256 O1s92"),
            "model15":
                ("None,64,None,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
                 "Ce3,3,32 Bn Ce3,3,48 Bn Rc Grs256 Grs256 Grs256 Grs256 "
                 "Grs256 O1s92"),
            "model16":
                ("None,64,None,1 Ce3,3,8 Bn Mp2,2,2,2 Ce3,3,12 Bn Ce3,3,20 Bn "
                 "Ce3,3,32 Bn Ce3,3,48 Bn Rc Gfs128 Gfs128 Gfs128 Gfs128 "
                 "Gfs128 O1s92"),
            "model17":
                ("None,64,None,1 Bn Ce3,3,16 RB3,3,16 RB3,3,16 RBd3,3,32 "
                 "RB3,3,32 RB3,3,32 RB3,3,32 RB3,3,32 RBd3,3,64 RB3,3,64 "
                 "RB3,3,64 RB3,3,64 RB3,3,64 RBd3,3,128 RB3,3,128 Rc Lr128 "
                 "Lr128 Lr128 Lr128 Lr128 O1s92")
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

        mapping = {'s': 'sigmoid', 't': 'tanh', 'r': 'relu',
                   'e': 'elu', 'l': None, 'm': 'softmax'}
        return mapping.get(nonlinearity, None)

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
        tf.layers.Input
            Input layer.
        """

        batch, height, width, depth = map(
            lambda x: None if x == "None" else int(x), inputs.split(","))

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
                         layer: str) -> tf.keras.layers.Conv2D:
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

        activation = self.get_activation_function(layer[1])
        conv_filter_params = [int(match)
                              for match in re.findall(r'\d+', layer)]
        if len(conv_filter_params) == 3:
            x, y, d = [int(match) for match in re.findall(r'\d+', layer)]
            logging.warning(
                "No stride provided, setting default stride of (1,1)")
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self._initializer)
        elif len(conv_filter_params) == 5:
            x, y, s_x, s_y, d = [int(match)
                                 for match in re.findall(r'\d+', layer)]
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(s_x, s_y),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self._initializer)
        else:
            raise ValueError(f"Conv layer {layer} not specified correctly")

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

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> maxpool_layer = vgsl_gn.maxpool_generator("Mp2,2,2,2")
        >>> type(maxpool_layer)
        <class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>
        """

        pool_x, pool_y, stride_x, stride_y = [
            int(match) for match in re.findall(r'\d+', layer)]
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

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> avgpool_layer = vgsl_gn.avgpool_generator("Ap2,2,2,2")
        >>> type(avgpool_layer)
        <class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>
        """

        pool_x, pool_y, stride_x, stride_y = [
            int(match) for match in re.findall(r'\d+', layer)]
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

        if layer[1] == 'c':
            prev_layer_y, prev_layer_x = prev_layer.shape[-2:]
            return layers.Reshape((-1, prev_layer_y * prev_layer_x))
        else:
            raise ValueError(f"Reshape layer {layer} not specified correctly")

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
            elu, or none.
            - `<d>`: Number of outputs.

        Returns
        -------
        tf.keras.layers.Dense
            A Dense layer with the specified parameters.

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

        activation, n = self.get_activation_function(
            layer[1]), int(re.search(r'\d+$', layer).group())
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
            `L(f|r|b)[s]<n>`
            - `(f|r|b)`: Direction of LSTM. 'f' for forward, 'r' for reversed,
            and 'b' for bidirectional.
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.

        Returns
        -------
        tf.keras.layers.LSTM
            An LSTM layer with the specified parameters.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> lstm_layer = vgsl_gn.lstm_generator("Lf64")
        >>> type(lstm_layer)
        <class 'keras.src.layers.rnn.lstm.LSTM'>
        """

        direction = layer[1]
        summarize = 's' in layer
        n = int(re.search(r'\d+$', layer).group())

        kwargs = {
            "units": n,
            "return_sequences": summarize,
            "go_backwards": direction == 'r',
            "kernel_initializer": self._initializer
        }

        rnn_layer = layers.LSTM
        return rnn_layer(**kwargs)

    def gru_generator(self,
                      layer: str) -> tf.keras.layers.GRU:
        """
        Generate a GRU layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the GRU layer. Expected format:
            `G(f|r|b)[s]<n>`
            - `(f|r|b)`: Direction of GRU. 'f' for forward, 'r' for reversed,
            and 'b' for bidirectional.
            - `[s]`: (Optional) Summarizes the output, outputting only the
            final step.
            - `<n>`: Number of outputs.

        Returns
        -------
        tf.keras.layers.Layer
            A GRU layer with the specified parameters.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> gru_layer = vgsl_gn.gru_generator("Gf64")
        >>> type(gru_layer)
        <class 'keras.src.layers.rnn.gru.GRU'>
        """

        direction = layer[1]
        summarize = 's' in layer
        n = int(re.search(r'\d+$', layer).group())

        kwargs = {
            "units": n,
            "return_sequences": summarize,
            "go_backwards": direction == 'r',
            "kernel_initializer": self._initializer
        }

        rnn_layer = layers.GRU
        return rnn_layer(**kwargs)

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
            `B(g|l)<n>`
            - `(g|l)`: Type of RNN layer. 'g' for GRU and 'l' for LSTM.
            - `<n>`: Number of units in the RNN layer.

        Returns
        -------
        tf.keras.layers.Bidirectional
            A Bidirectional RNN layer with the specified parameters.

        Raises
        ------
        ValueError
            If the RNN layer type specified in the VGSL string is not
            recognized.

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

        units_match = re.search(r'\d+$', layer)
        units = int(units_match.group())

        layer_type = layer[1]
        if layer_type == "l":
            rnn_layer = layers.LSTM
        elif layer_type == "g":
            rnn_layer = layers.GRU
        else:
            raise ValueError("Rnn layer type not found")

        rnn_params = {
            "units": units,
            "return_sequences": True,
            "kernel_initializer": self._initializer
        }

        return layers.Bidirectional(rnn_layer(**rnn_params),
                                    merge_mode='concat')

    def residual_block_generator(self,
                                 layer: str) -> ResidualBlock:
        """
        Generate a Residual Block based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the Residual Block. Expected format:
            `RB<x>,<y>,<d>`
            - `<x>`, `<y>`: Kernel sizes in the x and y dimensions
            respectively.
            - `<d>`: Depth of the Conv2D layers within the Residual Block.

        Returns
        -------
        ResidualBlock
            A Residual Block with the specified parameters.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> res_block = vgsl_gn.residual_block_generator("RB3,3,64")
        >>> type(res_block)
        <class 'custom_layers.ResidualBlock'>
        """

        downsample = 'd' in layer
        x, y, d = [int(match) for match in re.findall(r'\d+', layer)]
        return ResidualBlock(d, x, y, self._initializer, downsample)

    def dropout_generator(self,
                          layer: str) -> tf.keras.layers.Dropout:
        """
        Generate a Dropout layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the Dropout layer. Expected format:
            `Do<dropout>`
            - `<dropout>`: Dropout percentage (0-100).

        Returns
        -------
        tf.keras.layers.Dropout
            A Dropout layer with the specified dropout rate.

        Raises
        ------
        ValueError
            If the specified dropout rate is not in range [0, 100].

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> dropout_layer = vgsl_gn.dropout_generator("Do50")
        >>> type(dropout_layer)
        <class 'keras.src.layers.regularization.dropout.Dropout'>
        """

        dropout = int(re.search(r'\d+$', layer).group())
        if dropout < 0 or dropout > 100:
            raise ValueError("Dropout rate must be in range [0, 100]")

        return layers.Dropout(dropout/100)

    def get_output_layer(self,
                         layer: str,
                         output_classes: int = None) -> tf.keras.layers.Dense:
        """
        Generate an output layer based on a VGSL specification string.

        Parameters
        ----------
        layer : str
            VGSL specification for the output layer. Expected format:
            `O(2|1|0)(l|s|c)<n>`
            - `(2|1|0)`: Dimensionality of the output.
            - `(l|s|c)`: Non-linearity type.
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
            If the output layer type specified is not supported.

        Examples
        --------
        >>> vgsl_gn = VGSLModelGenerator("Some VGSL string")
        >>> output_layer = vgsl_gn.get_output_layer("O1s10", 10)
        >>> type(output_layer)
        <class 'keras.src.layers.core.dense.Dense'>
        """

        _, linearity, classes = layer[1], layer[2], int(
            re.search(r'\d+$', layer).group())

        if output_classes and classes != output_classes:
            logging.warning(
                "Overwriting output classes from input string. "
                "Was: %s, now: %s", classes, output_classes)
            classes = output_classes
            self.selected_model_vgsl_spec[-1] = f"O1{linearity}{classes}"

        if linearity == "s":
            return layers.Dense(classes,
                                activation='softmax',
                                kernel_initializer=self._initializer)
        elif linearity == "c":
            return CTCLayer(name='ctc_loss')

        else:
            raise ValueError("Output layer not yet supported")
