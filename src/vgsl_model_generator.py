# > Standard library
import re
import logging

# > Local dependencies
from custom_layers import CTCLayer, ResidualBlock

# > Third party dependencies
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, Input


class VGSLModelGenerator:
    """Nice description about this class I made some changes"""

    def __init__(self,
                 model,
                 name=None,
                 channels=None,
                 output_classes=None):
        super().__init__()
        self.initializer = initializers.GlorotNormal(seed=42)
        self.channel_axis = -1
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

    def init_model_from_string(self, vgsl_spec_string,
                               channels=None, output_classes=None):
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
                    axis=self.channel_axis))
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
                raise ValueError("The current layer: ", layer, " is not recognised, please check for correct formatting in the VGSL-Spec")

    def build(self):
        """ Loop through attributes and call forward pass """
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
    def get_model_libary():
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
                 "Cr3,3,96 Bn Cr3,3,96 Bn Mp2,2,2,2 Rc Lrs256 D20 Lrs256 D20 Lrs256 D20"
                 "Lrs256 D20 Lrs256 D20 O1s92"),
            "model11":
                ("None,64,None,1 Cr3,3,24 Bn Ap2,2,2,2 Cr3,3,48 Bn Cr3,3,96 Bn"
                 "Ap2,2,2,2 Cr3,3,96 Bn Ap2,2,2,2 Rc Grs256 Grs256 Grs256 Grs256 "
                 "Grs256 Fe1024 O1s92"),
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
                 "Ce3,3,32 Bn Ce3,3,48 Bn Rc Grs256 Grs256 Grs256 Grs256 Grs256 "
                 "O1s92"),
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
    def list_models():
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
    def get_units_or_outputs(layer):
        """ Retrieve the last digit from layer string"""
        match = re.findall(r'\d+', layer)
        return int(match[-1])

    @ staticmethod
    def get_activation_function(nonlinearity):
        """ Retrieve non-linearity function based on layer string """
        mapping = {'s': 'sigmoid', 't': 'tanh', 'r': 'relu',
                   'e': 'elu', 'l': None, 'm': 'softmax'}
        return mapping.get(nonlinearity, None)

    def make_input_layer(self, inputs, channels):
        """ Set Input from [batch, height, width, depth]"""
        batch, height, width, depth = map(
            lambda x: None if x == "None" else int(x), inputs.split(","))

        if channels and depth != channels:
            logging.warning("Overwriting channels from input string. "
                            "Was: %s, now: %s", depth, channels)
            depth = channels
            self.selected_model_vgsl_spec[0] = f"{batch},{height},{width},{depth}"

        logging.info("Creating input layer with shape: (%s, %s, %s, %s)",
                     batch, height, width, depth)
        return Input(shape=(width, height, depth), name='image')

    def get_input_shape(self):
        """ Retrieve the input_img layer for the init of the call() function"""
        return self.inputs.shape

    def get_vgsl_spec(self):
        """ Retrieve the vgls spec string set during the initialisation"""
        return self.selected_model_vgsl_spec

    #######################
    #   Layer functions   #
    #######################

    def conv2d_generator(self, layer):
        """
        C(s|t|r|l|m)<x>,<y>,<s_x>,<s_y>,<d> Convolves using a x,y window, with stride s_x, s_y and d units
        """
        activation = self.get_activation_function(layer[1])
        conv_filter_params = [int(match) for match in re.findall(r'\d+', layer)]
        if len(conv_filter_params) == 3:
            x, y, d = [int(match) for match in re.findall(r'\d+', layer)]
            logging.warning("No stride provided, setting default stride of (1,1)")
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(1, 1),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self.initializer)
        elif len(conv_filter_params) == 5:
            x, y, s_x, s_y, d = [int(match) for match in re.findall(r'\d+', layer)]
            return layers.Conv2D(d,
                                 kernel_size=(y, x),
                                 strides=(s_x, s_y),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=self.initializer)
        else:
            raise ValueError("Conv layer"+ layer +" not specified correctly")

    def maxpool_generator(self, layer):
        """ MaxPooling2D with pool_size and stride """
        pool_x, pool_y, stride_x, stride_y = [int(match) for match in re.findall(r'\d+', layer)]
        return layers.MaxPooling2D(pool_size=(pool_x, pool_y),
                                   strides=(stride_x, stride_y),
                                   padding='same')

    def avgpool_generator(self, layer):
        """ MaxPooling2D with pool_size and stride """
        pool_x, pool_y, stride_x, stride_y = [int(match) for match in re.findall(r'\d+', layer)]
        return layers.AvgPool2D(pool_size=(pool_x, pool_y),
                                strides=(stride_x, stride_y),
                                padding='same')

    def reshape_generator(self, layer, prev_layer):
        """
        Reshape output into new shape
        (c for collapse, i.e. -1, prev_layer_y*prev_layer_x)
        """
        if layer[1] == 'c':
            prev_layer_y, prev_layer_x = prev_layer.shape[-2:]
            return layers.Reshape((-1, prev_layer_y * prev_layer_x))
        else:
            return "Reshape operation not known"

    def fc_generator(self, layer):
        """
        F(s|t|r|l|m)<d> Fully-connected with s|t|r|l|m non-linearity and d
        outputs.
        Reduces height, width to 1. Connects to every y,x,depth position of the
        input, reducing height, width to 1, producing a single <d> vector as
        the output.
        Input height and width *must* be constant.
        For a sliding-window linear or non-linear map that connects just to the
        input depth, and leaves the input image size as-is, use a 1x1
        convolution (eg. Cr1,1,64 instead of Fr64).
        """
        activation, n = self.get_activation_function(layer[1]), int(re.search(r'\d+$', layer).group())
        return layers.Dense(n,
                            activation=activation,
                            kernel_initializer=self.initializer)

    def lstm_generator(self, layer):
        """
        L(f|r|b)[s]<n> LSTM cell with n outputs.
        The LSTM must have one of:
            f runs the LSTM forward only.
            r runs the LSTM reversed only.
        s (optional) summarizes the output in the requested dimension,
        outputting only the final step, collapsing the dimension to a single
        element.
        """
        direction = layer[1]
        summarize = 's' in layer
        n = int(re.search(r'\d+$', layer).group())

        kwargs = {
            "units": n,
            "return_sequences": summarize,
            "go_backwards": direction == 'r',
             "kernel_initializer": self.initializer
        }

        rnn_layer = layers.LSTM
        return rnn_layer(**kwargs)

    def gru_generator(self, layer):
        """
        G(f|r|b)[s]<n> GRU cell with n outputs
        """
        direction = layer[1]
        summarize = 's' in layer
        n = int(re.search(r'\d+$', layer).group())

        kwargs = {
            "units": n,
            "return_sequences": summarize,
            "go_backwards": direction == 'r',
             "kernel_initializer": self.initializer
        }

        rnn_layer = layers.GRU
        return rnn_layer(**kwargs)

    def bidirectional_generator(self, layer):
        """
        Create a Bidirectional layer with either a LSTM or GRU as the RNN layer
        Example Grs256 Bidirectional gru with 256 units
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
            "kernel_initializer": self.initializer
        }

        return layers.Bidirectional(rnn_layer(**rnn_params), merge_mode='concat')

    def residual_block_generator(self, layer):
        """
        Create a Residual Block with Conv2D layers and an elu BatchNorm, RB
        """
        downsample = 'd' in layer
        x, y, d = [int(match) for match in re.findall(r'\d+', layer)]
        return ResidualBlock(d, x, y, self.initializer, downsample)

    def dropout_generator(self, layer):
        """Dropout layer with probability"""
        dropout = int(re.search(r'\d+$', layer).group())
        if dropout >= 100:
            raise ValueError("Provided dropout is too high in " + layer + ", select a dropout < 100%")

        return layers.Dropout(dropout/100)

    def get_output_layer(self, layer, output_classes):
        """O(2|1|0)(l|s|c)n output layer with n classes.
          2 (heatmap) Output is a 2-d vector map of the input (possiLry at
            different scale). (Not yet supported.)
          1 (sequence) Output is a 1-d sequence of vector values.
          0 (category) Output is a 0-d single vector value.
          l uses a logistic non-linearity on the output, allowing multiple
            hot elements in any output vector value. (Not yet supported.)
          s uses a softmax non-linearity, with one-hot output in each value.
          c uses a softmax with CTC. Can only be used with s (sequence).
          NOTE Only O1s and O1c are currently supported. [01s for us]
        """
        _, linearity, classes = layer[1], layer[2], int(re.search(r'\d+$', layer).group())

        if output_classes and classes != output_classes:
            logging.warning("Overwriting output classes from input string. ""Was: %s, now: %s", classes, output_classes)
            classes = output_classes
            self.selected_model_vgsl_spec[-1] = f"O1{linearity}{classes}"

        if linearity == "s":
            return layers.Dense(classes,
                                activation='softmax',
                                kernel_initializer=self.initializer)
        elif linearity == "c":
            return CTCLayer(name='ctc_loss')

        else:
            raise ValueError("Output layer not yet supported")
