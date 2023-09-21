import os


from config import *
import utils

from data_loader import DataLoaderNew
from model import CERMetric, WERMetric, CTCLoss
from utils import *
import tensorflow.keras as keras

import numpy as np
import tensorflow as tf
import random
import argparse
from matplotlib import pyplot as plt
import tensorflow_addons as tfa

# disable GPU for now, because it is already running on my dev machine
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_DETERMINISTIC_OPS'] = '0'
from tensorflow.keras.utils import get_custom_objects

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.00001,
                    help='learning_rate to be used')
parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                    help='epochs to be used')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                    help='batch_size to be used, when using variable sized input this must be 1')

parser.add_argument('--height', metavar='height', type=int, default=64,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                    help='spec')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                    help='existing_model')
parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                    help='dataset. ecodices or iisg')
parser.add_argument('--do_binarize_otsu', action='store_true',
                    help='prefix to use for testing')
parser.add_argument('--do_binarize_sauvola', action='store_true',
                    help='do_binarize_sauvola')

parser.add_argument('--validation_list', metavar='validation_list', type=str, default=None,
                    help='validation_list')

args = parser.parse_args()

SEED = args.seed
GPU = args.gpu
PERCENT_VALIDATION = args.percent_validation
LEARNING_RATE = args.learning_rate
config.BATCH_SIZE = args.batch_size
config.EPOCHS = args.epochs
config.BASE_OUTPUT = args.output

MODEL_PATH = "../models/model-val-best/checkpoints/best_val/"
MODEL_PATH = "../model-republic-gru_mask-cer-0.02128436922850027"
MODEL_PATH = "../model-all-val_loss-22.38509"
MODEL_PATH = "../model-new7-128-batch32"

if args.existing_model:
    MODEL_PATH = args.existing_model

PLOT_PATH = os.path.sep.join([config.BASE_OUTPUT, "plot.png"])

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
config.IMG_SHAPE = (args.height, args.width, model_channels)
imgSize = config.IMG_SHAPE

model.summary()

layer_name = "conv3_block4_out"

submodel = model
print(submodel.summary())

model.summary()

partition = {'train': [], 'validation': [], 'test': []}
trainLabels = {}
valLabels = {}
testLabels = {}


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


def visualize_filter(filter_index, channels):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image(channels)
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


# 1 3 5 6
for layerId in range(len(submodel.layers)):
    layer = submodel.layers[layerId]
    if not layer.name.startswith("Conv") and not layer.name.startswith("add"):
        continue
    feature_extractor = keras.Model(inputs=submodel.inputs, outputs=layer.output)
    # feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

    all_imgs = []
    numFilters = layer.output_shape[3]
    # numFilters = 6
    i = 0
    for filter_index in range(numFilters):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index, model_channels)

        all_imgs.append(img)

    char_list = None
    # char_list_path = MODEL_PATH+"/charlist.txt"
    # with open(char_list_path) as f:
    #     char_list = f.readlines()

    maxTextLen = 128
    loader = DataLoaderNew(args.batch_size, imgSize,
                           train_list=None,
                           validation_list=None,
                           test_list=None,
                           inference_list=args.validation_list,
                           char_list=char_list,
                           check_missing_files=False
                           )

    training_generator, validation_generator, test_generator, inference_generator, utils, train_batches = loader.generators()

    inference_dataset = inference_generator

    batch_counter = 0
    for batch in inference_dataset:
        if batch_counter > 10:
            print('breaking')
            break
        item = batch[0]
        i = i + 1

        X = item
        maps = get_feature_maps(submodel, layerId, X[0])

        # Normalised [0,1]
        maps = (maps - np.min(maps)) / np.ptp(maps)
        maps = np.asarray(maps, dtype=np.float64)
        fig = plt.figure(figsize=(40, numFilters * 2))
        columns = 2
        rows = numFilters

        # ax enables access to manipulate each of subplots
        ax = []

        for j in range(numFilters):
            img = all_imgs[j - 1]
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, j * 2 + 1))
            ax[-1].set_title("ax:" + str(j))  # set title
            if model_channels == 1:
                img = tf.squeeze(img)
            plt.imshow(img)
            ax.append(fig.add_subplot(rows, columns, j * 2 + 2))
            # ax[-1].set_title("ax:" + str(j))  # set title
            if model_channels == 1:
                maps[j - 1] = tf.squeeze(maps[j - 1])
            # print(maps[j - 1]+0.5)
            plt.imshow(maps[j - 1]+0.5, cmap='gray')

        # plt.show()  # finally, render the plot

        # plt.show()
        filename = loader.get_item('inference', (batch_counter * args.batch_size))
        plt.tight_layout()
        plt.savefig('results/{}-{}'.format(layerId, os.path.basename(filename)))
        plt.close()
        batch_counter = batch_counter + 1
