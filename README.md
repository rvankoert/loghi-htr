# Loghi-core HTR

the source of this documentation is: https://github.com/knaw-huc/loghi-htr

```bash
# Install
sudo apt-get install python3.8-dev
git clone https://github.com/githubharald/CTCWordBeamSearch
cd CTCWordBeamSearch
python3.8 -m pip install .
cd ..

git clone https://github.com/knaw-huc/loghi-htr
cd htr/src/
python3.8 -m pip install -r requirements.txt
```


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

