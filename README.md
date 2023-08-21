# Loghi-core HTR

the source of this documentation is: https://github.com/rvankoert/htr

# Install
sudo apt-get install python3.8-dev
git clone https://github.com/githubharald/CTCWordBeamSearch
cd CTCWordBeamSearch
python3.8 -m pip install .

git clone https://github.com/rvankoert/htr
cd htr/src/
python3.8 -m pip install -r requirements.txt


# Usage
Loghi HTR is a system to generate text from images. It's part of the Loghi framework, which consists of several tools for layout analysis and HTR (Handwritten Text Recogntion).

Loghi HTR also works on machine printed text.

To run Loghi you need segmented textlines and their ground truth. You can generate ground truth from existing PageXML using tool MinionCutFromImageBasedOnPageXMLNew (Will be called LoghiLineCutter in the future) from the Loghi preprocessing or use existing linestrips provided by another tool.


Typical setup:
data -> textlines folder -> text line images 
data -> lines.txt (contains image location + transcription, tab separated)
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-0e54d043-4bab-40d7-9458-59aae79ed8a8.png	This is a ground truth transcription
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-f3d8b3cb-ab90-4b64-8360-d46a81ab1dbc.png	It can be generated from PageXML
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-700de0f9-56e9-45f9-a184-a4acbe6ed4cf.png	And another textline
...

for help run:
python3.8 main.py --help

to train using GPU use this line:
```
CUDA_VISIBLE_DEVICES=0 python3.8 main.py --do_train --train_list "training_all_ijsberg_na_train.txt training_all_globalise_train.txt" --validation_list "training_all_globalise_val.txt" --learning_rate 0.0001 --channels 4 --do_validate --gpu 0 --height 64 --use_mask --model newd14
``` 

--do_train: enable training stage
--train_list: list of files containing training data, must be of format: path_to_textline_image <TAB> transcription
--validation_list: list of files containing validation data, must be of format: path_to_textline_image <TAB> transcription
--learning_rate: learning rate to be used. Sane values range from 0.001 to 0.000001, 0.0003 being the default.
--channels: number of image channels to be used. Use 3 for standard RGB-images. 4 can be used for images where the alpha channel contains the polygon-mask of the textline
--do_validate: enable validation stage
--gpu: use -1 for CPU, 0 for the first GPU, 1 for the second GPU
--height: the height to scale the textline image to. All internal processing must be done on images of the same height. 64 is a sane value for handwriting
--use_mask: enable this when using batch_size > 1

on CPU:
CUDA_VISIBLE_DEVICES=-1 python3.8 main.py --do_train --train_list "training_all_ijsberg_na_train.txt training_all_globalise_train.txt" --validation_list "training_all_globalise_val.txt" --learning_rate 0.0001 --channels 4 --do_validate --gpu -1 --height 64 --use_mask 


to inference:
CUDA_VISIBLE_DEVICES=0 python3.8 /src/src/main.py --do_inference --channels 4 --height 64 --existing_model path_to_existing_model --batch_size 10 --use_mask --inference_list file_containing_lines_to_inferece.txt --results_file results.txt --charlist path_to_existing_model.charlist --gpu 0 --config_file_output config.txt --beam_width 10

During inferencing specific parameters must match those of the training phase: use_mask, height, channels


Docker images containing trained models are available via (to be inserted). Make sure to install nvidia-docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html


# [VGSL-Spec Manual](https://github.com/mldbai/tensorflow-models/blob/master/street/g3doc/vgslspecs.md)
**Disclaimer:** _The base models provided in the model_library were only tested on pre-processed HTR images with a height of 64 and variable width._  <br> <br>
Variable-size Graph Specification Language (VGSL) enables the specification of a TensorFlow graph, composed of convolutions and LSTMs, that can process variable-sized images, from a very short definition string, for example:

`[None,64,None,1 Cr3,3,32 Mp2,2,2,2 Cr3,3,64 Mp2,2,2,2 Rc Fc64 D20 Lrs128 D20 Lrs64 D20 O1s92]`

The user can decide the type of non-linearity function for certain specified layers using a lower-case character, for example `Cr3,3,32` uses a "Relu" activation function. Users can pick between `{'s': 'sigmoid', 't': 'tanh', 'r': 'relu',
                   'e': 'elu', 'l': 'linear', 'm': 'softmax'}`, denoted as `(s|t|r|e|l|m` for supported layers.

The current VGSL-Specs implementation supports the following layers:
## Overview Sheet

| **Layer**          | **Spec**                                       | **Example**      | **Description**                                                      |
|--------------------|------------------------------------------------|------------------|----------------------------------------------------------------------|
| Input              | `[batch, height, width, depth]`                | `None,64,None,1` | Input layer with variable batch_size & width, depth of 1 channel     |
| Output             | `O(2\|1\|0)(l\|s)`                             | `O1s10`          | Dense layer with a 1D sequence as with 10 output classes and softmax |
| Conv2D             | `C(s\|t\|r\|e\|l\|m),<x>,<y>[<s_x>,<s_y>],<d>` | `Cr,3,3,1,1`     | Conv2D layer with Relu, a 3x3 filter and 1,1 stride                  |
| Dense (FC)         | `F(s\|t\|r\|l\|m)<d>`                          | `Fs64`           | Dense layer with softmax and 64 units                                |
| LSTM               | `L(f\|r)[s]<n>`                                | `Lf64`           | Forward-only LSTM cell with 64 units                                 |
| GRU                | `G(f\|r)[s]<n>`                                | `Gr64`           | Reverse-only GRU cell with 64 units                                  |
| Bidirectional      | `B(g\|l)<n>`                                   | `Bl256`          | Bidirectional layer wrapping a LSTM RNN with 256 units               |
| BatchNormalization | `Bn`                                           | `Bn`             | BatchNormalization layer                                             |
| MaxPooling2D       | `Mp<x>,<y>,<s_x>,<s_y>`                        | `Mp2,2,1,1`      | MaxPooling2D layer with 2,2 pool size and 1,1 strides                |
| AvgPooling2D       | `Ap<x>,<y>,<s_x>,<s_y>`                        | `Ap2,2,2,2`      | AveragePooling2D layer with 2,2 pool size and 1,1 strides            |
| Dropout            | `D<rate>`                                      | `Do25`           | Dropout layer with `dropout` = 0.25                                  |
| ------------------ | ------------------------------------           |------------------|----------------------------------------------------------------------|
| ResidualBlock      | `TODO`                                         |                  |                                                                      |
| CTCLayer           | `TODO`                                         |                  |                                                                      |

## Layer Details
### Input:
<u>Spec</u>: **`[batch, height, width, depth]`** <br>
tf.keras.layers.Input layer, according to the standard TF tensor dimensions <br>
Example: `None,64,None,1` <br>
_Creates a tf.layers.Input with a variable batch size, height of 64, variable width and a depth of 1 (input channels)_
<br><br>  

### Output:
<u>Spec</u>: **`O(2|1|0)(l|s)<n>`** <br>
Output layer providing either a 2D vector (heat) map of the input (`2`), a 1D sequence of vector values (`1`) or a 0D single vector value (`0`). Currently, only `1` is supported. with `n` classes <br>
Example: `O1s10` <br>
_Creates a Dense layer with a 1D sequence as output with 10 classes and softmax_
<br><br>  

### Conv2D:
<u>Spec</u>: **`C(s|t|r|e|l|m)<x>,<y>,[<s_x>,<s_y>,]<d>`** <br>
Convolves using a `x`,`y` window and `d` units. Optionally, the stride window can be set with (`s_x`, `s_y`) <br>
Example: `Cr3,3,1,1` <br>
_Creates a Conv2D layer with a Relu activation function a 3x3 filter and 1,1 stride (if s_x and s_y are not provided, set to (1,1) default)_
<br><br>  

### **Dense (Fully-connected layer)**:
<u>Spec</u>: **`F(s|t|r|e|l|m)<d>`** <br>
Fully-connected(FC)) with `s|t|r|e|l|m` non-linearity and `d` outputs <br>
Example: `Fs64` <br>
_Creates a FC layer with softmax non-linearity and 64 outputs_
<br><br>  

### **LSTM**:
<u>Spec</u>: **`L(f|r)[s]<n>`** <br>
LSTM cell running either forward-only (`f`) or reversed-only (`r`), with `n` outputs. Optionally, summarization of the output at the final step can be added by providing a Boolean `d` (corresponds with the return_sequences)<br>
Example: `Lf64` <br>
_Creates a forward-only LSTM cell with 64 outputs_
<br><br>  

### **GRU**:
<u>Spec</u>: **`G(f|r)[s]<n>`** <br>
GRU cell running either forward-only (`f`) or reversed-only (`r`), with `n` outputs. Optionally, summarization of the output at the final step can be added by providing a Boolean `d` (corresponds with the return_sequences)<br>
Example: `Gf64` <br>
_Creates a forward-only GRU cell with 64 outputs_
<br><br>  

### **Bidirectional**:
<u>Spec</u>: **`B(g|l)<n>`** <br>
Bidirectional layer which wraps either a LSTM (`l`) or GRU (`g`) RNN layer and runs it in both forward and backward directions. <br>
Example: `Bl256` <br>
_Creates a Bidirectional RNN layer using a LSTM Cell with 256 outputs_
<br><br>  

### **BatchNormalization**:
<u>Spec</u>: **`Bn`** <br>
A technique often used to standardize the inputs to a layer for each mini-batch. Has stabilizing properties during the learning process
and reduces the number of epochs required to train DNNs. <br>
    Example: `Bn` <br>
Applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1 ([source](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization))
<br><br>  

### **MaxPooling2D**:
Downsampling technique along the height and width of a 2D image by taking the maximum value over the input window `x`,`y` and is shifted by strides along each dimension (`s_x`, `s_y`). `padding` is always set to "same".<br>
    <u>Spec</u>: **`Mp<x>,<y>,<s_x>,<s_y>`** <br>
    Example: `Mp2,2,2,2` <br>
    _Creates a MaxPooling2D layer with pool size (2,2) and strides of (2,2)_
<br><br>

### **AvgPooling2D**:
Downsampling technique along the height and width of a 2D image by taking the average value over the input window x,y and is shifted by strides along each dimension (s_x, s_y). `padding` is always set to "same".<br>
    <u>Spec</u>: **`Ap<x>,<y>,<s_x>,<s_y>`** <br>
    Example: `Ap2,2,2,2` <br>
    _Creates a AveragePooling2D layer with pool size (2,2) and strides of (2,2)_
<br><br>  

### **Dropout**:
Regularization layer that randomly sets input units to 0 with a frequency of `rate` at each step during training time. Used to prevent overfitting.    
<u>Spec</u>: **`D<rate>`** <br>
    Example: `Do50` <br>
    _Creates a Dropout layer with a dropout rate of 0.5 (`D`/100)_
<br><br> 

---

### Custom blocks:
* **ResidualBlock**: TODO
<br><br>

* **CTCLayer**: TODO
<br><br>
