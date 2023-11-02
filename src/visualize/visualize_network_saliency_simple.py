# Imports

# > Standard Library
import os
import sys
import random

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from vis_arg_parser import get_args
from model import CERMetric, WERMetric, CTCLoss

# > Third party libraries
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import cv2

# disable GPU for now, because it is already running on my dev machine
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '0'

args = get_args()

if args.existing_model:
    if not os.path.exists(args.existing_model):
        print('cannot find existing model on disk: ' + args.existing_model)
        exit(1)
    MODEL_PATH = args.existing_model
else:
    print('Please provide a path to an existing model directory')
    exit(1)

SEED = args.seed
GPU = args.gpu

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
if GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if (len(gpus) > 0):
        tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

print("[INFO] loading dataset...")

get_custom_objects().update({"CERMetric": CERMetric})
get_custom_objects().update({"WERMetric": WERMetric})
get_custom_objects().update({"CTCLoss": CTCLoss})

model = keras.models.load_model(MODEL_PATH)
model_channels = model.input_shape[3]

print("model_channels:",model_channels)
submodel = model
print(submodel.summary())

if args.sample_image:
    if not os.path.exists(args.sample_image):
        print('cannot find validation .txt on disk: ' + args.sample_image)
        exit(1)
    IMG_PATH = args.sample_image
else:
    print(
        'Please provide a path to a .txt containing (validation) images with the --sample_image parameter, e.g. the sample_list.txt inside "loghi-htr/tests/data" ')
    exit(1)

print("model channels:",model_channels)

#Remake data_generator parts
target_height = 64
original_image = tf.io.read_file(IMG_PATH)
original_image = tf.image.decode_png(original_image, channels=model_channels)
original_image = tf.image.resize(original_image,
                        [target_height,
                         tf.cast(target_height * tf.shape(original_image)[1]
                                 / tf.shape(original_image)[0], tf.int32)],
                        preserve_aspect_ratio=True)

tf.keras.utils.save_img("test_img.png",original_image)
image_width = tf.shape(original_image)[1]
print("img width shape:",image_width)
image_height = tf.shape(original_image)[0]
original_image = tf.image.resize_with_pad(original_image, target_height, image_width + 50)
tf.keras.utils.save_img("test_img_padded.png", original_image)
# Normalize the image and something else
img = 0.5 - (original_image / 255)
tf.keras.utils.save_img("test_img_padded_normalised.png", original_image)
img = tf.transpose(img, perm=[1, 0, 2])
img = np.expand_dims(img, axis=0)

print(img.shape)
preds = model.predict(img)
preds = tf.dtypes.cast(preds,tf.float32)
charlist = MODEL_PATH + "charlist.txt"
with open(charlist,'r') as f:
    charlist = f.read()

print(charlist)
print("last char in charlist:",charlist[-1])

import sys
np.set_printoptions(threshold=sys.maxsize)

timestep_charlist_indices = []
timestep_charlist_indices_top_5 = []
top_k = 3
for text_line in preds:
    timesteps = len(text_line)
    step_width_unpadded = tf.get_static_value(image_width) / timesteps
    step_width = tf.get_static_value(image_width+50) / timesteps
    pad_steps_skip = np.floor(50/step_width)
    print("Step width: ", step_width)
    print("Time steps: ", timesteps)
    for time_step in text_line:
        timestep_charlist_indices.append(tf.get_static_value(tf.math.argmax(time_step)))
        timestep_charlist_indices_top_5.append(tf.get_static_value(tf.math.top_k(time_step,k=top_k,sorted=True)))

print(len(charlist))
# with open(MODEL_PATH + "charlist.txt",'r') as f:
#     array_charlist = list(char for char in f.read())
# print(len(array_charlist))

# for i in timestep_charlist_indices_top_5:
#     print(i)
#     for x in i.indices:
#         if x == len(charlist)+1:
#             print("BLANK")
#         else:
#             print(charlist[x-1])

def remove_tags(text):
    text = text.replace('␃', '') #     public static String STRIKETHROUGHCHAR = "␃"; //Unicode Character “␃” (U+2403)
    text = text.replace('␅', '') #    public static String UNDERLINECHAR = "␅"; //Unicode Character “␅” (U+2405)
    text = text.replace('␄', '') #    public static String SUBSCRIPTCHAR = "␄"; // Unicode Character “␄” (U+2404)
    text = text.replace('␆', '') #    public static String SUPERSCRIPTCHAR = "␆"; // Unicode Character “␆” (U+2406)
    return text

original_image = cv2.imread(IMG_PATH,cv2.IMREAD_UNCHANGED)
original_image_padded = cv2.resize(original_image,(tf.get_static_value(image_width+50),64))
cv2.imwrite("original_image_padded.png",original_image_padded)

bordered_img = cv2.copyMakeBorder(original_image_padded,50,200,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
cv2.putText(bordered_img, "Time-step predictions (Top-1 prediction):",
            org=(0, 15),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2)

timestep_char_labels_cleaned = []
line_start = 25

print("BLANK TOKEN IN CHARLIST?:",'' in charlist)
# if blank token in charlist -> take normal else i-1

if '' in charlist:
    index_correction = 0
else:
    index_correction = -1

#Retrieve the top 5 predictions for each timestep and write them down underneath each other
for index, char_index in enumerate(timestep_charlist_indices):
    if index < pad_steps_skip:
        continue
    line_start += step_width
    start_point = (int(line_start), 50)
    end_point = (int(line_start), tf.get_static_value(image_height) + 50)
    if char_index < len(charlist)+1: # charlist + 1 is blank token, which is final character
        timestep_char_labels_cleaned.append(remove_tags(charlist[char_index + index_correction]))
        if charlist[char_index] == " ": # Do not draw predictions that are spaces for better readability
            continue
        cv2.line(bordered_img,
                 start_point,
                 end_point,
                 color =(0,0,255),
                 thickness=1)
        cv2.putText(bordered_img,
                    remove_tags(charlist[char_index + index_correction]),
                    org=start_point,
                    color=(0,0,255),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    thickness=1)

        #Add the top-2 to top-5 most probable characters for the current time step
        table_start_height = 170
        for top_char_index in timestep_charlist_indices_top_5[index].indices.numpy()[1:]:
            if top_char_index < len(charlist)+1:
                cv2.putText(bordered_img,
                            remove_tags(charlist[top_char_index + index_correction]),
                            org = (int(line_start),table_start_height),
                            color = (0,0,255),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            thickness=1)
                table_start_height += 50

#Add the top-5 for the specific timestep
row_num = 2
min_row_y = 170
max_row_y = 170 + ((top_k-1) * 50)
for row_height in range(170,max_row_y,50):
    #Draw the top-5 text
    cv2.putText(bordered_img, "Top-" + str(row_num),
                org=(0, row_height),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                thickness=1)
    row_num += 1

#Add misc. text
cv2.putText(bordered_img,"Other predictions for this time step (less probable):",
            org=(0, 140),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            thickness=1)
cv2.putText(bordered_img, "Final result (collapsed blank characters):",
            org=(0, 270),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            thickness=1)

#Add the cleaned version of the prediction
cv2.putText(bordered_img,
            "".join(timestep_char_labels_cleaned),
            org=(50, 300),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            thickness=1)
# print line and character in image using line_start
print("".join(timestep_char_labels_cleaned))
cv2.imwrite("final_img.jpg", bordered_img)

#Add the visualize_network filters at the bottom
one_string = tf.strings.format("{}\n", (preds),summarize=-1)
tf.io.write_file("output.txt",one_string)
