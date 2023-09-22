# Loghi-core HTR

Loghi HTR is a system to generate text from images. It's part of the Loghi framework, which consists of several tools for layout analysis and HTR (Handwritten Text Recogntion).

Loghi HTR also works on machine printed text.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Variable-size Graph Specification Language (VGSL)](#variable-size-graph-specification-language-vgsl)

## Installation

This section provides a step-by-step guide to installing Loghi HTR and its dependencies.

### Prerequisites

Ensure you have the following prerequisites installed or set up:

- Ubuntu or a similar Linux-based operating system. The provided commands are tailored for such systems.

### Steps

1. **Install Python 3**

```bash
sudo apt-get install python3
```

2. **Clone and install CTCWordBeamSearch**

```bash
git clone https://github.com/githubharald/CTCWordBeamSearch
cd CTCWordBeamSearch
python3 -m pip install .
```

3. **Clone the HTR repository and install its requirements**

```bash
git clone https://github.com/rvankoert/loghi-htr.git
cd loghi-htr
python3 -m pip install -r requirements.txt
```

With these steps, you should have Loghi HTR and all its dependencies installed and ready to use.

## Usage

### Setting Up

1. **(Optional) Organize Text Line Images**

    While not mandatory, for better organization, you can place your text line images in a 'textlines' folder or any desired location. The crucial point is that the paths mentioned in 'lines.txt' should be valid and point to the respective images.

2. **Generate a 'lines.txt' File**

    This file should contain the locations of the image files and their respective transcriptions. Separate each location and transcription with a tab.

Example of 'lines.txt' content:

```
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-0e54d043-4bab-40d7-9458-59aae79ed8a8.png	This is a ground truth transcription
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-f3d8b3cb-ab90-4b64-8360-d46a81ab1dbc.png	It can be generated from PageXML
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-700de0f9-56e9-45f9-a184-a4acbe6ed4cf.png	And another textline
```

### Command-Line Options:

The command-line options include, but are not limited to:

- `--do_train`: Enable the training stage.
- `--do_validate`: Enable the validation stage.
- `--do_inference`: Perform inference.
- `--train_list`: List of files containing training data. Format: `path/to/textline/image <TAB> transcription`.
- `--validation_list`: List of files containing validation data. Format: `path/to/textline/image <TAB> transcription`.
- `--inference_list`: List of files containing data to perform inference on. Format: `path/to/textline/image`.
- `--learning_rate`: Set the learning rate. Recommended values range from 0.001 to 0.000001, with 0.0003 being the default.
- `--channels`: Number of image channels. Use 3 for standard RGB-images, and 4 for images with an alpha channel containing the textline polygon-mask.
- `--gpu`: GPU configuration. Use -1 for CPU, 0 for the first GPU, and so on.
- `--batch_size`: The number of examples to use as input in the model at the same time. Increasing this requires more RAM or VRAM.
- `--height`: Height to scale the textline image. Internal processing requires images of the same height. 64 is recommended for handwriting.
- `--use_mask`: Enable when using `batch_size` > 1.
- `--results_file`: The inference results are aggregated in this file.
- `--config_file_output`: The output location of the config.

For detailed options and configurations, you can also refer to the help command:

```bash
python3 main.py --help
```

### Usage Examples

**Note**: Ensure that the value of `CUDA_VISIBLE_DEVICES` matches the value provided to `--gpu`. For instance, if you set `CUDA_VISIBLE_DEVICES=0`, then `--gpu` should also be set to 0. If you explicitely want to use CPU (not recommended), make sure to set this value to -1.

**Training on GPU**

```bash
CUDA_VISIBLE_DEVICES=0 
python3 main.py 
--model model14
--do_train 
--train_list "train_lines_1.txt train_lines_2.txt" 
--do_validate 
--validation_list "validation_lines_1.txt" 
--height 64 
--channels 4 
--learning_rate 0.0001 
--use_mask 
--gpu 0 
```

**Inference on GPU**

```bash
CUDA_VISIBLE_DEVICES=0 
python3 main.py 
--existing_model /path/to/existing/model 
--charlist /path/to/existing/model/charlist.txt
--do_inference 
--inference_list "inference_lines_1.txt"
--height 64 
--channels 4 
--beam_width 10
--use_mask 
--gpu 0 
--batch_size 10 
--results_file results.txt 
--config_file_output config.txt 
```

_Note_: During inferencing, certain parameters, such as use_mask, height, and channels, must match the parameters used during the training phase.

### Typical setup


Docker images containing trained models are available via (to be inserted). Make sure to install nvidia-docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


## Variable-size Graph Specification Language (VGSL)

Variable-size Graph Specification Language (VGSL) is a powerful tool that enables the creation of TensorFlow graphs, comprising convolutions and LSTMs, tailored for variable-sized images. This concise definition string simplifies the process of defining complex neural network architectures. For a detailed overview of VGSL, also refer to the [official documentation](https://github.com/mldbai/tensorflow-models/blob/master/street/g3doc/vgslspecs.md).

**Disclaimer:** _The base models provided in the `VGSLModelGenerator.model_library` were only tested on pre-processed HTR images with a height of 64 and variable width._

### How VGSL works

VGSL operates through short definition strings. For instance:

`[None,64,None,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc Fc64 D20 Lrs128 D20 Lrs64 D20 O1s92]`

In this example, the string defines a neural network with input layers, convolutional layers, pooling, reshaping, fully connected layers, LSTM and output layers. Each segment of the string corresponds to a specific layer or operation in the neural network. Moreover, VGSL provides the flexibility to specify the type of activation function for certain layers, enhancing customization.

### Supported Layers and Their Specifications

| **Layer**          | **Spec**                                       | **Example**        | **Description**                                                                                              |
|--------------------|------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------|
| Input              | `[batch, height, width, depth]`                | `None,64,None,1`   | Input layer with variable batch_size & width, depth of 1 channel                                             |
| Output             | `O(2\|1\|0)(l\|s)`                             | `O1s10`            | Dense layer with a 1D sequence as with 10 output classes and softmax                                         |
| Conv2D             | `C(s\|t\|r\|e\|l\|m),<x>,<y>[<s_x>,<s_y>],<d>` | `Cr,3,3,64`        | Conv2D layer with Relu, a 3x3 filter, 1x1 stride and 64 filters                                              |
| Dense (FC)         | `F(s\|t\|r\|l\|m)<d>`                          | `Fs64`             | Dense layer with softmax and 64 units                                                                        |
| LSTM               | `L(f\|r)[s]<n>`                                | `Lf64`             | Forward-only LSTM cell with 64 units                                                                         |
| GRU                | `G(f\|r)[s]<n>`                                | `Gr64`             | Reverse-only GRU cell with 64 units                                                                          |
| Bidirectional      | `B(g\|l)<n>`                                   | `Bl256`            | Bidirectional layer wrapping a LSTM RNN with 256 units                                                       |
| BatchNormalization | `Bn`                                           | `Bn`               | BatchNormalization layer                                                                                     |
| MaxPooling2D       | `Mp<x>,<y>,<s_x>,<s_y>`                        | `Mp2,2,1,1`        | MaxPooling2D layer with 2x2 pool size and 1x1 strides                                                        |
| AvgPooling2D       | `Ap<x>,<y>,<s_x>,<s_y>`                        | `Ap2,2,2,2`        | AveragePooling2D layer with 2x2 pool size and 1x1 strides                                                    |
| Dropout            | `D<rate>`                                      | `Do25`             | Dropout layer with `dropout` = 0.25                                                                          |
| Reshape            | `Rc`                                           | `Rc`               | Reshape layer returns a new (collapsed) tf.Tensor with a different shape based on the previous layer outputs |
| ResidualBlock      | **TODO**                                       | **TODO**           | **TODO**                                                                                                     |
| CTCLayer           | **TODO**                                       | **TODO**           | **TODO**                                                                                                     |

### Layer Details
#### Input

- **Spec**: `[batch, height, width, depth]`
- **Description**: Represents the input layer in TensorFlow, based on standard TF tensor dimensions.
- **Example**: `None,64,None,1` creates a tf.layers.Input with a variable batch size, height of 64, variable width and a depth of 1 (input channels)

#### Output

- **Spec**: `O(2|1|0)(l|s)<n>`
- **Description**: Output layer providing either a 2D vector (heat) map of the input (`2`), a 1D sequence of vector values (`1`) or a 0D single vector value (`0`) with `n` classes. Currently, only a 1D sequence of vector values is supported. 
- **Example**: `O1s10` creates a Dense layer with a 1D sequence as output with 10 classes and softmax.

#### Conv2D

- **Spec**: `C(s|t|r|e|l|m)<x>,<y>,[<s_x>,<s_y>],<d>`
- **Description**: Convolutional layer using a `x`,`y` window and `d` filters. Optionally, the stride window can be set with (`s_x`, `s_y`).
- **Examples**: 
  - `Cr3,3,64` creates a Conv2D layer with a Relu activation function, a 3x3 filter, 1x1 stride, and 64 filters.
  - `Cr3,3,1,3,128` creates a Conv2D layer with a Relu activation function, a 3x3 filter, 1x3 strides, and 128 filters.

#### Dense (Fully-connected layer)

- **Spec**: `F(s|t|r|e|l|m)<d>`
- **Description**: Fully-connected layer with `s|t|r|e|l|m` non-linearity and `d` units.
- **Example**: `Fs64` creates a FC layer with softmax non-linearity and 64 units.

#### LSTM

- **Spec**: `L(f|r)[s]<n>`
- **Description**: LSTM cell running either forward-only (`f`) or reversed-only (`r`), with `n` units.
- **Example**: `Lf64` creates a forward-only LSTM cell with 64 units.

#### GRU

- **Spec**: `G(f|r)[s]<n>`
- **Description**: GRU cell running either forward-only (`f`) or reversed-only (`r`), with `n` units.
- **Example**: `Gf64` creates a forward-only GRU cell with 64 units.

#### Bidirectional

- **Spec**: `B(g|l)<n>`
- **Description**: Bidirectional layer wrapping either a LSTM (`l`) or GRU (`g`) RNN layer, running in both directions, with `n` units.
- **Example**: `Bl256` creates a Bidirectional RNN layer using a LSTM Cell with 256 units.

#### BatchNormalization

- **Spec**: `Bn`
- **Description**: A technique often used to standardize the inputs to a layer for each mini-batch. Helps stabilize the learning process.
- **Example**: `Bn` applies a transformation maintaining mean output close to 0 and output standard deviation close to 1.

#### MaxPooling2D

- **Spec**: `Mp<x>,<y>,<s_x>,<s_y>`
- **Description**: Downsampling technique using a `x`,`y` window. The window is shifted by strides `s_x`, `s_y`.
- **Example**: `Mp2,2,2,2` creates a MaxPooling2D layer with pool size (2,2) and strides of (2,2).

#### AvgPooling2D

- **Spec**: `Ap<x>,<y>,<s_x>,<s_y>`
- **Description**: Downsampling technique using a `x`,`y` window. The window is shifted by strides `s_x`, `s_y`.
- **Example**: `Ap2,2,2,2` creates an AveragePooling2D layer with pool size (2,2) and strides of (2,2).

#### Dropout

- **Spec**: `D<rate>`
- **Description**: Regularization layer that sets input units to 0 at a rate of `rate` during training. Used to prevent overfitting.
- **Example**: `Do50` creates a Dropout layer with a dropout rate of 0.5 (`D`/100).

#### Reshape

- **Spec**: `Rc`
- **Description**: Reshapes the output tensor from the previous layer, making it compatible with RNN layers.
- **Example**: `Rc` applies a specific transformation: `layers.Reshape((-1, prev_layer_y * prev_layer_x))`.

---

#### Custom blocks:
- **ResidualBlock**: Documentation in progress.
- **CTCLayer**: Documentation in progress.

