# Loghi

Loghi is a set of tools for Handwritten Text Recognition. 

Two sample scripts are provided to make starting everything a little bit easier. 
na-pipeline.sh: for transcribing scans
na-pipeline-train.sh: for training new models. 

## Quick start

Install Loghi so that you can use its pipeline script.
```bash
git clone git@github.com:knaw-huc/loghi.git
cd loghi
```

## Use the docker images
The easiest method to run Loghi is to use the default dockers images on [Docker Hub](https://hub.docker.com/u/loghi).
The docker images are usually pulled automatically when running [`na-pipeline.sh`](na-pipeline.sh) mentioned later in this document, but you can pull them separately with the following commands:

```bash
docker pull loghi/docker.laypa
docker pull loghi/docker.htr
docker pull loghi/docker.loghi-tooling
```

If you do not have Docker installed follow [these instructions](https://docs.docker.com/engine/install/) to install it on your local machine.

If you instead want to build the dockers yourself with the latest code:
```bash
git submodule update --init --recursive
cd docker
./buildAll.sh
=======
git clone https://github.com/knaw-huc/loghi-htr.git
cd loghi-htr
python3 -m pip install -r requirements.txt

```
This also allows you to have a look at the source code inside the dockers. The source code is available in the submodules.


## Inference

But first go to:
https://surfdrive.surf.nl/files/index.php/s/YA8HJuukIUKznSP
and download a laypa model (for detection of baselines) and a loghi-htr model (for HTR).

suggestion for laypa:
- general

suggestion for loghi-htr that should give some results:
- generic-2023-02-15

It is not perfect, but a good starting point. It should work ok on 17th and 18th century handwritten dutch. For best results always finetune on your own specific data.

edit the [`na-pipeline.sh`](na-pipeline.sh) using vi, nano, other whatever editor you prefer. We'll use nano in this example

```bash
nano na-pipeline.sh
```
Look for the following lines:
```
LAYPAMODEL=INSERT_FULL_PATH_TO_YAML_HERE
LAYPAMODELWEIGHTS=INSERT_FULLPATH_TO_PTH_HERE
HTRLOGHIMODEL=INSERT_FULL_PATH_TO_LOGHI_HTR_MODEL_HERE
```
and update those paths with the location of the files you just downloaded. If you downloaded a zip: you should unzip it first.

if you do not have a NVIDIA-GPU and nvidia-docker setup additionally change

```text
GPU=0
```
to
```text
GPU=-1
```
It will then run on CPU, which will be very slow. If you are using the pretrained model and run on CPU: please make sure to download the Loghi-htr model starting with "float32-". This will run faster on CPU than the default mixed_float16 models.


Save the file and run it:
```bash
./na-pipeline.sh /PATH_TO_FOLDER_CONTAINING_IMAGES
```
replace /PATH_TO_FOLDER_CONTAINING_IMAGES with a valid directory containing images (.jpg is preferred/tested) directly below it.

The file should run for a short while if you have a good nvidia GPU and nvidia-docker setup. It might be a long while if you just have CPU available. It should work either way, just a lot slower on CPU.
=======
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

When it finishes without errors a new folder called "page" should be created in the directory with the images. This contains the PageXML output.

## Training an HTR model

### Input data

Expected structure
```text
training_data_folder
|- training_all_train.txt
|- training_all_val.txt
|- image1_snippets
    |-snippet1.png
    |-snippet2.png
```

`training_all_train.txt` should look something something like:
```text
/path/to/training_data_folder/image1_snippets/snippet1.png	textual representation of snippet 1
/path/to/training_data_folder/image1_snippets//snippet2.png text on snippet 2
```
n.b. path to image and textual representation should be separated by a tab.

##### Create training data
You can create training data with the following command:
```bash
./create_train_data.sh /full/path/to/input /full/path/to/output
```
`/full/path/to/output` is `/full/path/to/training_data_folder` in this example
`/full/path/to/input` is expected to look like:
```text
input
|- image1.png
|- image2.png
|- page
    |- image1.xml
    |- image2.xml
```
`page/image1.xml` should contain information about the baselines and should have the textual representation of the text lines.  

### Change script
Edit the [`na-pipeline-train.sh`](na-pipeline-train.sh) script using your favorite editor:

```bash
nano na-pipeline-train.sh
```

Find the following lines:
```text
listdir=INSERT_FULL_PATH_TO_TRAINING_DATA_FOLDER
trainlist=INSERT_FULL_PATH_TO_TRAINING_DATA_LIST
validationlist=INSERT_FULL_PATH_TO_VALIDATION_DATA_LIST
```
In this example: 
```text
listdir=/full/path/to/training_data_folder
trainlist=/full/path/to/training_data_folder/train_list.txt
validationlist=/full/path/to/training_data_folder/val_list.txt
```

if you do not have a NVIDIA-GPU and nvidia-docker setup additionally change:

```text
GPU=0
```
to
```text
GPU=-1
```
It will then run on CPU, which will be very slow.


### Run script
Finally, to run the HTR training run the script:

```bash
./na-pipeline-train.sh
```

## For later updates use:
To update the submodules to the head of their branch (the latest/possibly unstable version) run the following command:
```bash
git submodule update --recursive --remote
```
