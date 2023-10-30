# Imports

# > Standard Library
import os
import sys
import random

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from data_loader import DataLoader
from utils import initialize_image, deprocess_image, ctc_decode, Utils
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
model_channels = model.layers[0].input_shape[0][3]
img_size = (args.height, args.width, model_channels)
submodel = model
print(submodel.summary())

char_list = None
if args.sample_image:
    if not os.path.exists(args.sample_image):
        print('cannot find validation .txt on disk: ' + args.sample_image)
        exit(1)
    IMG_PATH = args.sample_image
else:
    print(
        'Please provide a path to a .txt containing (validation) images with the --sample_image parameter, e.g. the sample_list.txt inside "loghi-htr/tests/data" ')
    exit(1)

print("model channesl:",model_channels)
sample_image = initialize_image(model_channels)
img = cv2.imread(IMG_PATH)

#Remake data_generator parts
original_image = tf.io.read_file(IMG_PATH)
original_image = tf.image.decode_png(original_image, channels=model_channels)
original_image = tf.image.resize(original_image, (64, 99999), preserve_aspect_ratio=True) / 255.0
tf.keras.utils.save_img("test_img.png",original_image)
image_width = tf.shape(original_image)[1]
print("img width shape:",image_width)
image_height = tf.shape(original_image)[0]
original_image = tf.image.resize_with_pad(original_image, 64, image_width + 50)
tf.keras.utils.save_img("test_img_padded.png", original_image)
img = 0.5 - original_image
tf.keras.utils.save_img("test_img_padded_normalised.png", original_image)
img = tf.transpose(img, perm=[1, 0, 2])
img = np.expand_dims(img, axis=0)

#Remove last layer softmax
preds = model.predict(img)
preds = tf.dtypes.cast(preds,tf.float32)
charlist = MODEL_PATH + "charlist.txt"
with open(charlist,'r') as f:
    charlist = f.read()

import sys
np.set_printoptions(threshold=sys.maxsize)

timestep_charlist_indices = []
for text_line in preds:
    timesteps = len(text_line)
    step_width_unpadded = tf.get_static_value(image_width) / timesteps
    step_width = tf.get_static_value(image_width+50) / timesteps
    pad_steps_skip = np.floor(50/step_width)
    print("Step width: ", step_width)
    print("Time steps: ", timesteps)
    for time_step in text_line:
        timestep_charlist_indices.append(tf.get_static_value(tf.math.argmax(time_step)))

print(timestep_charlist_indices)
timestep_char_labels_cleaned = []
linestart = 25

original_image = cv2.imread(IMG_PATH,cv2.IMREAD_UNCHANGED)
original_image_padded = cv2.resize(original_image,(tf.get_static_value(image_width+50),64))
cv2.imwrite("original_image_padded.png",original_image_padded)

bordered_img = cv2.copyMakeBorder(original_image_padded,50,50,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
cv2.putText(bordered_img, "Time-step predictions:",
            org=(0, 15),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2)

for index, char_index in enumerate(timestep_charlist_indices):
    index += 1
    if index < pad_steps_skip:
        continue
    linestart += step_width
    start_point = (int(linestart), 50)
    end_point = (int(linestart), tf.get_static_value(image_height)+50)
    if char_index < len(charlist)+1 : # charlist + 1 is blank token, which is final character
        timestep_char_labels_cleaned.append(charlist[char_index])
        cv2.line(bordered_img,
                 start_point,
                 end_point,
                 color =(0,0,255),
                 thickness=1)
        cv2.putText(bordered_img,charlist[char_index],
                    org=(int(linestart),50),
                    color=(0,0,255),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    thickness=1)

#Add the clean version of the prediction
cv2.putText(bordered_img, "Cleaned prediction:",
            org=(0, 130),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2)

cv2.putText(bordered_img,"".join(timestep_char_labels_cleaned),
            org=(50, 159),
            color=(0, 0, 0),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            thickness=1)
# print line and character in image using linestart
print("".join(timestep_char_labels_cleaned))
cv2.imwrite("final_img.jpg",bordered_img)

#Add the visualize_network filters at the bottom
one_string = tf.strings.format("{}\n", (preds),summarize=-1)
tf.io.write_file("output.txt",one_string)

# if __name__ == '__main__':
