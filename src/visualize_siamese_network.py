import math
import os

import utils
from Model import CTCLoss, CERMetric, WERMetric
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from config import *
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import random
import argparse
import tensorflow_addons as tfa
from keras.utils.generic_utils import get_custom_objects


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

# @tf.function
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
    iterations = 20
    learning_rate = 20.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = utils.deprocess_image(img[0].numpy())
    return loss, img


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--height', metavar='height', type=int, default=51,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=4,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                    help='existing_model')

args = parser.parse_args()

SEED = args.seed
GPU = args.gpu
config.IMG_SHAPE = (args.height, args.width, args.channels)
config.BASE_OUTPUT = args.output

MODEL_PATH = "../models/model-val-best/checkpoints/best_val/"
MODEL_PATH = "../model-republic-gru_mask-cer-0.02128436922850027"
MODEL_PATH = "../model-all-val_loss-22.38509"
MODEL_PATH = "../model-new7-128-batch32"


if args.existing_model:
    MODEL_PATH = args.existing_model

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

# model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
model = keras.models.load_model(MODEL_PATH)
model.summary()
config.IMG_SHAPE = (128, 128, args.channels)
layer_name = "conv3_block4_out"
# submodel=model
submodel = model
print(submodel.summary())
for layer in submodel.layers:
    if not layer.name.startswith("Conv") and not layer.name.startswith("add"):
        continue
    # print(layer.name)
    # continue
    feature_extractor = keras.Model(inputs=submodel.inputs, outputs=layer.output)
    # feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

    all_imgs = []
    numFilters = layer.output_shape[3]
    # numFilters = 8
    for filter_index in range(numFilters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = math.ceil(math.sqrt(numFilters))
    cropped_width = config.IMG_SHAPE[0] - 25 * 2
    cropped_height = config.IMG_SHAPE[1]- 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, args.channels))

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

