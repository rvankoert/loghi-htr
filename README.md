# Loghi HTR

Loghi HTR is a system to generate text from images. It's part of the Loghi framework, which consists of several tools for layout analysis and HTR (Handwritten Text Recogntion).

Loghi HTR also works on machine printed text.

To run Loghi you need segmented textlines and their ground truth. You can generate ground truth from existing PageXML using tool XXX from the Loghi preprocessing or use existing linestrips provided by another tool.


Typical setup:
data -> textlines folder -> text line images 
data -> lines.txt (contains image location + transcription, tab separated)
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-0e54d043-4bab-40d7-9458-59aae79ed8a8.png	This is a ground truth transcription
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-f3d8b3cb-ab90-4b64-8360-d46a81ab1dbc.png	It can be generated from PageXML
/data/textlines/NL-HaNA_2.22.24_HCA30-1049_0004/NL-HaNA_2.22.24_HCA30-1049_0004.xml-700de0f9-56e9-45f9-a184-a4acbe6ed4cf.png	And another textline
...

for help run:
python3.8 main.py --help

to train use this line:
CUDA_VISIBLE_DEVICES=0 python3.8 main.py --do_train --train_list "training_all_ijsberg_na_train.txt training_all_globalise_train.txt" --validation_list "training_all_globalise_val.txt" --learning_rate 0.0001 --channels 4 --batch_size 24 --epochs 100 --do_validate --gpu 0 --height 64 --memory_limit 11000 --seed 2 --model new8 --beam_width 1 --use_mask --rnn_layers 3 --decay_steps 50000 --rnn_units 256 --existing_model ../model-new8-ijsberg_republicrandom_prizepapers_64_val_loss_5.6246
