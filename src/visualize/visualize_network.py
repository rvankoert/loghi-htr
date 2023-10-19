# Imports

# > Standard Library
import math
import os
import random
import sys

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from model import CERMetric, WERMetric, CTCLoss
from utils import *
from config import *
from vis_arg_parser import get_args

# > Third party libraries
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import get_custom_objects

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '0'

def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def initialize_image():
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, config.IMG_SHAPE[0], config.IMG_SHAPE[1], config.IMG_SHAPE[2]))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img -0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
    img = tf.transpose(img[0].numpy(), perm=[1, 0, 2])
    # Decode the resulting input image
    img = deprocess_image(img)
    return loss, img

args = get_args()

if args.existing_model:
    if not os.path.exists(args.existing_model):
        print('cannot find existing model on disk: ' + args.existing_model)
        exit(1)
    MODEL_PATH = args.existing_model

SEED = args.seed
GPU = args.gpu
config.BASE_OUTPUT = args.output
PLOT_PATH = os.path.sep.join([config.BASE_OUTPUT, "plot.png"])

if not os.path.exists(config.BASE_OUTPUT):
    os.makedirs(config.BASE_OUTPUT)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
if GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

imgSize = config.IMG_SHAPE
keras.losses.custom_loss = tfa.losses.contrastive_loss
get_custom_objects().update({"CERMetric": CERMetric})
get_custom_objects().update({"WERMetric": WERMetric})
get_custom_objects().update({"CTCLoss": CTCLoss})

model = keras.models.load_model(MODEL_PATH)
model_channels = model.layers[0].input_shape[0][3]

model.summary()
config.IMG_SHAPE = (64, 64, model_channels)
layer_name = "conv3_block4_out"
submodel = model
print(submodel.summary())
for layer in submodel.layers:
    if not layer.name.startswith("Conv") and not layer.name.startswith("conv") and not layer.name.startswith("add"):
        continue
    feature_extractor = keras.Model(inputs=submodel.inputs, outputs=layer.output)

    all_imgs = []
    numFilters = layer.output_shape[3]
    for filter_index in range(numFilters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = math.ceil(math.sqrt(numFilters))
    cropped_width = config.IMG_SHAPE[0] #- 25 * 2
    cropped_height = config.IMG_SHAPE[1] #- 25 * 2
    width = n * cropped_width + (n - 1) * margin * 2
    height = n * cropped_height + (n - 1) * margin * 2
    stitched_filters = np.zeros((width, height, model_channels))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            if i * n + j >= numFilters:
                break
            print(len(all_imgs))
            img = all_imgs[i * n + j]
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    keras.preprocessing.image.save_img("{}/stitched_filters_{}.png".format(config.BASE_OUTPUT, layer.name), stitched_filters)

