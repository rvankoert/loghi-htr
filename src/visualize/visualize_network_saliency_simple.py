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
# img = deprocess_image(sample_image)
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
# img = img.reshape((args.height, args.width, args.channels))
# with open(IMG_PATH.replace(".png",".txt"),'r') as f:
#     img_label = f.read()

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size, color_mode="grayscale")
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

print(img.shape)
img_array = get_img_array(IMG_PATH, size=(img.shape[1],64))

#Remake data_generator parts
original_image = tf.io.read_file(IMG_PATH)
original_image = tf.image.decode_png(original_image, channels=4)
original_image = tf.image.resize(original_image, (64, 99999), preserve_aspect_ratio=True) / 255.0
tf.keras.utils.save_img("test_img.png",original_image)
image_width = tf.shape(original_image)[1]
image_height = tf.shape(original_image)[0]
original_image = tf.image.resize_with_pad(original_image, 64, image_width + 50)
tf.keras.utils.save_img("test_img_padded.png", original_image)
img = 0.5 - original_image
tf.keras.utils.save_img("test_img_padded_normalised.png", original_image)
img = tf.transpose(img, perm=[1, 0, 2])
tf.keras.utils.save_img("test_img_padded_transposed.png", img)
img = np.expand_dims(img, axis=0)
tf.keras.utils.save_img("test_img_padded_transposed_expanded.png", img.squeeze())

#Remove last layer softmax
# model.layers[-1].activation = tf.keras.activations.linear
# print(img)
preds = model.predict(img)
preds = tf.dtypes.cast(preds,tf.float32)
charlist = MODEL_PATH + "charlist.txt"
with open(charlist,'r') as f:
    charlist = f.read()

print(charlist)

utilsObject = Utils(charlist,False)

#Decode predictions so we know the output?
print(preds.shape)

print(len(charlist))
print("last char", charlist[-1])
print("last char -2 ", charlist[-2])
print(preds.shape[0] * preds.shape[1])
top_paths= 1
output_texts = []

import sys
np.set_printoptions(threshold=sys.maxsize)

#List of lists of lists[0]

timestep_charlist_indices = []
for text_line in preds:
    timesteps = len(text_line)
    step_width = tf.get_static_value(image_width) / timesteps
    print("Step width: ", step_width)
    print("Time steps: ", timesteps)
    for time_step in text_line:
        timestep_charlist_indices.append(tf.get_static_value(tf.math.argmax(time_step)))

print(timestep_charlist_indices)
timestep_char_labels_cleaned = []
linestart = 0

import matplotlib.pyplot as plt
fig = plt.figure()

#Remove batch dimension
print(img.shape)

squeezed_img = tf.transpose(img, perm=[0,2,1,3])
print(squeezed_img.shape)
tf.keras.utils.save_img("squeezed_img.png", np.squeeze(squeezed_img))
# print(squeezed_img)
squeezed_img = np.squeeze(squeezed_img)

for char_index in timestep_charlist_indices:
    linestart += step_width
    start_point = (int(linestart), tf.get_static_value(image_height))
    end_point = (int(linestart + step_width), tf.get_static_value(image_height))
    if char_index < len(charlist)+1: # charlist + 1 is blank token, which is final character
        timestep_char_labels_cleaned.append(charlist[char_index])

# img = cv2.line(original_image, start_point, end_point, color =(255,0,0), thickness=1)
    cv2.line(squeezed_img, start_point, end_point, color =(255,0,0), thickness=5)

# print(squeezed_img)


        # print line and character in image using linestart

print("".join(timestep_char_labels_cleaned))
tf.keras.utils.save_img("squeezed_lines.png", squeezed_img)
# cv2.imshow("lines",img)
# cv2.imwrite("test_img.png",img)
# tf.io.write_file("new_img.png",img)

# write the image


# for pred in preds:
    # print("pred parsing:")
    # for label_index in pred:
        # print(label_index)
        # print(charlist[label_index])
        # print(charlist[label_index-2])

one_string = tf.strings.format("{}\n", (preds),summarize=-1)
tf.io.write_file("output.txt",one_string)



# print(pred_labels)
# print(pred_labels[0].tolist())



# for pred in pred_labels:
#     # print("pred parsing:")
#     for label_index in pred:
#         # print(label_index)
#         print(charlist[label_index])
#         print(charlist[label_index-2])

# pred_labels_chars = [charlist[x-1] if x is not len(pred_labels[0].tolist())-1 else "BLANK" for x in pred_labels[0].tolist()]
# print(pred_labels_chars)
# pred_labels_chars = [charlist.]



# print("Predicted:", decode_predictions(preds, top=1)[0])


#
#
#
# # @tf.function
# print([img_label])
#
# saliency = compute_saliency(tf.convert_to_tensor(img), tf.convert_to_tensor([img_label]))
# # Compute max absolute gradient
# saliency = tf.reduce_max(tf.abs(saliency), axis=-1)  def compute_saliency(image, label):
#     with tf.GradientTape() as tape:
#         tape.watch(image)
#         prediction = model(image)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
#
#     gradients = tape.gradient(loss, image)
#     return gradients
# saliency /= saliency.numpy().max()  # Normalize to [0, 1]
#
# plt.figure(figsize=(8, 8))
# plt.imshow(sample_image[0], cmap='gray')
# plt.imshow(saliency[0], cmap='jet', alpha=0.6)
# plt.colorbar()
# plt.title('Saliency Map')
# plt.axis('off')
# plt.show()
