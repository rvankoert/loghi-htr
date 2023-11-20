# Loghi-core HTR

Loghi HTR is a system to generate text from images. It's part of the Loghi framework, which consists of several tools for layout analysis and HTR (Handwritten Text Recogntion).

Loghi HTR also works on machine printed text.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Variable-size Graph Specification Language (VGSL)](#variable-size-graph-specification-language-vgsl)
4. [API Usage Guide](#api-usage-guide)
5. [Model Visualizer Guide](#model-visualizer-guide)
5. [Frequently Asked Questions (FAQ)](#FAQ)

## Installation

This section provides a step-by-step guide to installing Loghi HTR and its dependencies.

### Prerequisites

Ensure you have the following prerequisites installed or set up:

- Ubuntu or a similar Linux-based operating system. The provided commands are tailored for such systems.

> [!IMPORTANT]
> The requirements listed in `requirements.txt` require a Python version > 3.8. It should be possible to run in Python <= 3.8, but one would have to downgrade some packages (such as NumPy and Tensorflow).

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
- `--train_list`: List of files containing training data. Format: `/path/to/textline/image <TAB> transcription`.
- `--validation_list`: List of files containing validation data. Format: `/path/to/textline/image <TAB> transcription`.
- `--inference_list`: List of files containing data to perform inference on. Format: `/path/to/textline/image`.
- `--learning_rate`: Set the learning rate. Recommended values range from 0.001 to 0.000001, with 0.0003 being the default.
- `--channels`: Number of image channels. Use 3 for standard RGB-images, and 4 for images with an alpha channel containing the textline polygon-mask.
- `--gpu`: GPU configuration. Use -1 for CPU, 0 for the first GPU, and so on.
- `--batch_size`: The number of examples to use as input in the model at the same time. Increasing this requires more RAM or VRAM.
- `--height`: Height to scale the textline image. Internal processing requires images of the same height. 64 is recommended for handwriting.
- `--use_mask`: Enable when using `batch_size` > 1.
- `--results_file`: The inference results are aggregated in this file.
- `--config_file_output`: The output location of the config.
- `--replace_recurrent_layer`: Specifies the [VGSL string](#variable-size-graph-specification-language-vgsl) to define the architecture of the recurrent layers that will replace the recurrent layers of an existing model. This argument is required when you want to modify the recurrent layers of a model specified by `--existing_model`. The VGSL string describes the type, direction, and number of units for the recurrent layers. For example, "Lfs128 Lf64" describes two LSTM layers with 128 and 64 units respectively. When using this argument, ensure that `--existing_model` is also provided to specify the model whose recurrent layers you want to replace.
- `--replace_final_layer`: Enables the replacement of the final dense layer of an existing model. This is useful when you want to adjust the number of output characters or change the masking option without modifying the rest of the model. When this argument is used, the model specified by `--existing_model` will have its final dense layer replaced with a new one. The number of output units will be adjusted based on the value of `number_characters` and whether the `--use_mask` option is enabled.

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

`None,64,None,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc Fc64 D20 Lrs128 D20 Lrs64 D20 O1s92`

In this example, the string defines a neural network with input layers, convolutional layers, pooling, reshaping, fully connected layers, LSTM and output layers. Each segment of the string corresponds to a specific layer or operation in the neural network. Moreover, VGSL provides the flexibility to specify the type of activation function for certain layers, enhancing customization.

### Supported Layers and Their Specifications

| **Layer**          | **Spec**                                       | **Example**        | **Description**                                                                                              |
|--------------------|------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------|
| Input              | `batch,height,width,depth`                    | `None,64,None,1`   | Input layer with variable batch_size & width, depth of 1 channel                                             |
| Output             | `O(2\|1\|0)(l\|s)`                             | `O1s10`            | Dense layer with a 1D sequence as with 10 output classes and softmax                                         |
| Conv2D             | `C(s\|t\|r\|e\|l\|m),<x>,<y>[<s_x>,<s_y>],<d>` | `Cr3,3,64`        | Conv2D layer with Relu, a 3x3 filter, 1x1 stride and 64 filters                                              |
| Dense (FC)         | `F(s\|t\|r\|l\|m)<d>`                          | `Fs64`             | Dense layer with softmax and 64 units                                                                        |
| LSTM               | `L(f\|r)[s]<n>,[D<rate>,Rd<rate>]`             | `Lf64`             | Forward-only LSTM cell with 64 units                                                                         |
| GRU                | `G(f\|r)[s]<n>,[D<rate>,Rd<rate>]`             | `Gr64`             | Reverse-only GRU cell with 64 units                                                                          |
| Bidirectional      | `B(g\|l)<n>[D<rate>Rd<rate>]`                  | `Bl256`            | Bidirectional layer wrapping a LSTM RNN with 256 units                                                       |
| BatchNormalization | `Bn`                                           | `Bn`               | BatchNormalization layer                                                                                     |
| MaxPooling2D       | `Mp<x>,<y>,<s_x>,<s_y>`                        | `Mp2,2,1,1`        | MaxPooling2D layer with 2x2 pool size and 1x1 strides                                                        |
| AvgPooling2D       | `Ap<x>,<y>,<s_x>,<s_y>`                        | `Ap2,2,2,2`        | AveragePooling2D layer with 2x2 pool size and 2x2 strides                                                    |
| Dropout            | `D<rate>`                                      | `D25`             | Dropout layer with `dropout` = 0.25                                                                          |
| Reshape            | `Rc`                                           | `Rc`               | Reshape layer returns a new (collapsed) tf.Tensor with a different shape based on the previous layer outputs |
| ResidualBlock      | `RB[d]<x>,<y>,<z>`                             | `RB3,3,64`         | Residual Block with optional downsample. Has a kernel size of <x>,<y> and a depth of <z>. If `d` is provided, the block will downsample the input |

### Layer Details
#### Input

- **Spec**: `batch,height,width,depth`
- **Description**: Represents the input layer in TensorFlow, based on standard TF tensor dimensions.
- **Example**: `None,64,None,1` creates a `tf.layers.Input` with a variable batch size, height of 64, variable width and a depth of 1 (input channels)

#### Output

- **Spec**: `O(2|1|0)(l|s)<n>`
- **Description**: Output layer providing either a 2D vector (heat) map of the input (`2`), a 1D sequence of vector values (`1`) or a 0D single vector value (`0`) with `n` classes. Currently, only a 1D sequence of vector values is supported. 
- **Example**: `O1s10` creates a Dense layer with a 1D sequence as output with 10 classes and softmax.

#### Conv2D

- **Spec**: `C(s|t|r|e|l|m)<x>,<y>[,<s_x>,<s_y>],<d>`
- **Description**: Convolutional layer using a `x`,`y` window and `d` filters. Optionally, the stride window can be set with (`s_x`, `s_y`).
- **Examples**: 
  - `Cr3,3,64` creates a Conv2D layer with a Relu activation function, a 3x3 filter, 1x1 stride, and 64 filters.
  - `Cr3,3,1,3,128` creates a Conv2D layer with a Relu activation function, a 3x3 filter, 1x3 strides, and 128 filters.

#### Dense (Fully-connected layer)

- **Spec**: `F(s|t|r|e|l|m)<d>`
- **Description**: Fully-connected layer with `s|t|r|e|l|m` non-linearity and `d` units.
- **Example**: `Fs64` creates a FC layer with softmax non-linearity and 64 units.

#### LSTM

- **Spec**: `L(f|r)[s]<n>[,D<rate>,Rd<rate>]`
- **Description**: LSTM cell running either forward-only (`f`) or reversed-only (`r`), with `n` units. Optionally, the `rate` can be set for the `dropout` and/or the `recurrent_dropout`, where `rate` indicates a percentage between 0 and 100.
- **Example**: `Lf64` creates a forward-only LSTM cell with 64 units.

#### GRU

- **Spec**: `G(f|r)[s]<n>[,D<rate>,Rd<rate>]`
- **Description**: GRU cell running either forward-only (`f`) or reversed-only (`r`), with `n` units. Optionally, the `rate` can be set for the `dropout` and/or the `recurrent_dropout`, where `rate` indicates a percentage between 0 and 100.
- **Example**: `Gf64` creates a forward-only GRU cell with 64 units.

#### Bidirectional

- **Spec**: `B(g|l)<n>[,D<rate>,Rd<rate>]`
  - **Description**: Bidirectional layer wrapping either a LSTM (`l`) or GRU (`g`) RNN layer, running in both directions, with `n` units. Optionally, the `rate` can be set for the `dropout` and/or the `recurrent_dropout`, where `rate` indicates a percentage between 0 and 100.
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
- **Example**: `D50` creates a Dropout layer with a dropout rate of 0.5 (`D`/100).

#### Reshape

- **Spec**: `Rc`
- **Description**: Reshapes the output tensor from the previous layer, making it compatible with RNN layers.
- **Example**: `Rc` applies a specific transformation: `layers.Reshape((-1, prev_layer_y * prev_layer_x))`.

#### ResidualBlock
- **Spec**: `RB[d]<x>,<y>,<z>`
- **Description**: A Residual Block with a kernel size of <x>,<y> and a depth of <z>. If [d] is provided, the block will downsample the input. Residual blocks are used to allow for deeper networks by adding skip connections, which helps in preventing the vanishing gradient problem.
- **Example**: `RB3,3,64` creates a Residual Block with a 3x3 kernel size and a depth of 64 filters.

## API Usage Guide

This guide walks you through the process of setting up and running the API, as well as how to interact with it.

### 1. Setting up the API

Navigate to the `src/api` directory in your project:

```bash
cd src/api
```

#### Starting the API

You have the choice to run the API using either `gunicorn` (recommended) or `flask`. To start the server:

Using `gunicorn`:

```bash
python3 gunicorn_app.py
```

Or using `flask`:

```bash
python3 flask_app.py
```

#### Environment Variables Configuration

Before running the app, you must set several environment variables. The app fetches configurations from these variables:

**Gunicorn Options:**

```bash
GUNICORN_RUN_HOST        # Default: "127.0.0.1:8000": The host and port where the API should run.
GUNICORN_WORKERS         # Default: "1": Number of worker processes.
GUNICORN_THREADS         # Default: "1": Number of threads per worker.
GUNICORN_ACCESSLOG       # Default: "-": Access log settings.
```

**Loghi-HTR Options:**

```bash
LOGHI_MODEL_PATH         # Path to the model.
LOGHI_CHARLIST_PATH      # Path to the character list.
LOGHI_MODEL_CHANNELS     # Number of channels in the model.
LOGHI_BATCH_SIZE         # Default: "256": Batch size for processing.
LOGHI_OUTPUT_PATH        # Directory where predictions are saved.
LOGHI_MAX_QUEUE_SIZE     # Default: "10000": Maximum size of the processing queue.
```

**GPU Options:**

```bash
LOGHI_GPUS               # Default: "0": GPU configuration.
```

You can set these variables in your shell or use a script. An example script to start a `gunicorn` server can be found in `src/api/start_local_app.sh`.

### 2. Interacting with the running API

Once the API is up and running, you can send HTR requests using curl. Here's how:

```bash
curl -X POST -F "image=@$input_path" -F "group_id=$group_id" -F "identifier=$filename" http://localhost:5000/predict
```

Replace `$input_path`, `$group_id`, and `$filename` with your specific values. The model processes the image, predicts the handwritten text, and saves the predictions in the specified output path (from the `LOGHI_OUTPUT_PATH` environment variable).

---

This guide should help you get started with the API. For advanced configurations or troubleshooting, please reach out for support.

## Model Visualizer Guide

The following instructions will explain how to generate visualizations that can help describe an existing model's learned representations when provided with a sample image. The visualizer requires a trained model and a sample image (e.g. PNG or JPG):

<figure> <img src="https://raw.githubusercontent.com/rvankoert/loghi-htr/visualize-files-revamp/src/visualize/visualize_plots/sample_image.jpg" alt="sample_image" width="650" style="display: block; margin: 0 auto;" /> <figcaption>Example time-step prediction </figure>

### 1. Visualize setup
Navigate to the `src/visualize` directory in your project:

```bash
cd src/visualize
```

### 2. Start the visualizers
```bash
python3 main.py 
--existing_model /path/to/existing/model 
--sample_image /path/to/sample/img
```
This will output various files into the `visualize_plots directory`:
* A PDF sheet consisting of all made visualizations for the above call
* Individual PNG and JPG files of these visualizations
* a `sample_image_preds.csv` which consist of a character prediction table for each prediction timestep. The highest probability is the character that was chosen by the model

Currently, the following visualizers are implemented:

1. **visualize_timestep_predictions**: Takes the `sample_image` and simulates the model's prediction process for each time step, the top-3 most probable characters per timestep are displayed and the "cleaned" result is shown at the bottom.

2. **visualize_filter_activations**: Display what the convolutional filters have learned after providing it with random noise + show the activation of conv filters for the `sample_image`. Each unique convolutional layer is displayed once.

Potential future implementations:
* Implement a SHAP visualizer to show the parts of the image that influence the model's character prediction. Or a similar saliency plot.
* Plot the raw Conv filters (e.g. a 3x3 filter)

**Note**:  If a model has multiple `Cr3,3,64` layers then only the first instance of this configuration is visualized)


### 3. (Optional parameters)
```bash
--do_detailed # Visualize all convolutional layers, not just the first instance of a conv layer
--light_mode  # Plots and overviews are shown in light mode (instead of dark mode)
```

---

## FAQ

If you're new to using this tool or encounter issues, this FAQ section provides answers to common questions and problems. If you don't find your answer here, please reach out for further assistance.

### How can I determine the VGSL spec of a model I previously used?

If you've used one of our older models and would like to know its VGSL specification, follow these steps:

**For Docker users:**

1. If your Docker container isn't already running with the model directory mounted, start it and bind mount your model directory:

```bash
docker run -it -v /path/on/host/to/your/model_directory:/path/in/container/to/model_directory loghi/docker.htr
```

Replace `/path/on/host/to/your/model_directory` with the path to your model directory on your host machine, and `/path/in/container/to/model_directory` with the path where you want to access it inside the container.

2. Once inside the container, run the VGSL spec generator:

```bash
python3 /src/loghi-htr/src/vgsl_model_generator.py --model_dir /path/in/container/to/model_directory
```

Replace `/path/in/container/to/model_directory` with the path you specified in the previous step.

**For Python users:**

1. Run the VGSL spec generator:

```bash
python3 src/vgsl_model_generator.py --model_dir /path/to/your/model_directory
```

Replace `/path/to/your/model_directory` with the path to the directory containing your saved model.

### How do I use `replace_recurrent_layer`?

The `replace_recurrent_layer` is a feature that allows you to replace the recurrent layers of an existing model with a new architecture defined by a VGSL string. To use it:

1. Specify the model you want to modify using the `--existing_model` argument.
2. Provide the VGSL string that defines the new recurrent layer architecture with the `--replace_recurrent_layer` argument. The VGSL string describes the type, direction, and number of units for the recurrent layers. For example, "Lfs128 Lfs64" describes two LSTM layers with 128 and 64 units respectively, with both layers returning sequences.
3. (Optional) Use `--use_mask` if you want the replaced layer to account for masking.
4. Execute your script or command, and the tool will replace the recurrent layers of your existing model based on the VGSL string you provided.

### I'm getting the following error when I want to use `replace_recurrent_layer`: `Input 0 of layer "lstm_1" is incompatible with the layer: expected ndim=3, found ndim=2.` What do I do?

This error usually indicates that there is a mismatch in the expected input dimensions of the LSTM layer. Often, this is because the VGSL spec for the recurrent layers is missing the `[s]` argument, which signifies that the layer should return sequences.

To resolve this:
- Ensure that your VGSL string for the LSTM layer has an `s` in it, which will make the layer return sequences. For instance, instead of "Lf128", use "Lfs128".
- Re-run the script or command with the corrected VGSL string.
