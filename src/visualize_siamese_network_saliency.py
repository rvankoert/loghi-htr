import metrics
from datagenerator import DataGenerator
from config import *
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import random
import argparse
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tensorflow.keras.preprocessing.image import load_img
from keras.utils.generic_utils import get_custom_objects

from dataset_ecodices import DatasetEcodices
from dataset_iisg import DatasetIISG
from dataset_medieval import DatasetMedieval
from dataset_medieval_30percent import DatasetMedieval30Percent
from dataset_medieval_30percent_sample import DatasetMedieval30PercentSample
from dataset_place_century_script import DatasetPlaceCenturyScript

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

parser.add_argument('--height', metavar='height', type=int, default=51,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=3,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--trainset', metavar='trainset', type=str, default='/data/cvl-database-1-1/train.txt',
                    help='trainset to be used')
parser.add_argument('--testset', metavar='testset', type=str, default='/data/cvl-database-1-1/test.txt',
                    help='testset to be used')
parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                    help='testset to be used')
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

args = parser.parse_args()

SEED = args.seed
GPU = args.gpu
PERCENT_VALIDATION = args.percent_validation
LEARNING_RATE = args.learning_rate
config.IMG_SHAPE = (args.height, args.width, args.channels)
config.BATCH_SIZE = args.batch_size
config.EPOCHS = args.epochs
config.BASE_OUTPUT = args.output

MODEL_PATH = os.path.sep.join([config.BASE_OUTPUT, "siamese_model"])
MODEL_PATH = "checkpoints/difornet13-saved-model-68-0.94.hdf5"
MODEL_PATH = "checkpoints/difornet13-saved-model-49-0.94.hdf5"  # iisg
MODEL_PATH = "checkpoints/difornet14-saved-model-45-0.97.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet17-saved-model-44-0.92.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-98-0.97.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-128-0.95.hdf5"
MODEL_PATH = "checkpoints/difornet23-best_val_loss"
MODEL_PATH = "checkpoints/difornet24-best_val_loss"
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

imgSize = config.IMG_SHAPE
print("[INFO] loading DiFor dataset...")

# model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
# model = keras.models.load_model(MODEL_PATH)
# keras.losses.custom_loss = metrics.contrastive_loss

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"average": metrics.average})
model = keras.models.load_model(MODEL_PATH)
# model = keras.models.load_model('/home/rutger/src/siamesenew/output.92percent/siamese_model/')

model.summary()

# config.IMG_SHAPE = (180, 180, 3)
layer_name = "conv3_block4_out"

submodel = model.get_layer(index=2)
submodel.summary()

# Load images
# img1 = load_img('images/goldfish.jpg')
img2 = load_img('images/bear.jpg')

if args.dataset == 'iisg':
    training_generator, validation_generator, test_generator = DatasetIISG().generators(args.channels,
                                                                                        args.do_binarize_otsu,
                                                                                        args.do_binarize_sauvola)
if args.dataset == 'ecodices':
    training_generator, validation_generator, test_generator = DatasetEcodices().generators(args.channels,
                                                                                            args.do_binarize_otsu,
                                                                                            args.do_binarize_sauvola)
if args.dataset == 'medieval':
    training_generator, validation_generator, test_generator = DatasetMedieval().generators(args.channels,
                                                                                            args.do_binarize_otsu,
                                                                                            args.do_binarize_sauvola)
if args.dataset == 'medieval_small':
    training_generator, validation_generator, test_generator = DatasetMedieval30Percent().generators(args.channels,
                                                                                                     args.do_binarize_otsu,
                                                                                                     args.do_binarize_sauvola)
if args.dataset == 'medieval_small_sample':
    training_generator, validation_generator, test_generator = DatasetMedieval30PercentSample().generators(
        args.channels, args.do_binarize_otsu)
if args.dataset == 'place_century_script':
    training_generator, validation_generator, test_generator = DatasetPlaceCenturyScript().generators(args.channels,
                                                                                                      args.do_binarize_otsu,
                                                                                                      args.do_binarize_sauvola)

i = 0


def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    # return (output[0][1], output[1][1], output[2][1])
    return output[0][0]


def model_modifier(model):
    model.layers[-1].activation = tf.keras.activations.linear
    return model


while i < 100:
    item = test_generator.__getitem__(i)

    # put sample into list
    i = i + 1

    X = item
    # Rendering

    saliency = Saliency(model,
                        model_modifier=model_modifier,
                        clone=False)
    predicted = model.predict(X[0])
    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map2 = saliency(loss,
                             X[0],
                             smooth_samples=20,  # The number of calculating gradients iterations.
                             smooth_noise=0.20)  # noise spread level.
    saliency_map1 = normalize(saliency_map2[0])
    saliency_map2 = normalize(saliency_map2[1])

    subplot_args = {'nrows': 4, 'ncols': 1, 'figsize': (18, 6),
                    'subplot_kw': {'xticks': [], 'yticks': []}}
    f, ax = plt.subplots(**subplot_args)
    title = "same"
    if X[1] == 0:
        title = "different"
    if predicted[0][0] < 0.5:
        title = title + " same"
    else:
        title = title + " different"
    ax[0].set_title(title, fontsize=14)
    img1 = tf.keras.utils.array_to_img(K.squeeze(X[0][0], axis=-0))
    ax[0].imshow(img1)
    ax[1].imshow(saliency_map1[0], cmap='jet')
    img2 = tf.keras.utils.array_to_img(K.squeeze(X[0][1], axis=-0))
    ax[2].set_title(predicted[0][0], fontsize=14)
    ax[2].imshow(img2)
    ax[3].imshow(saliency_map2[0], cmap='jet')
    plt.tight_layout()
    plt.savefig('results-saliency/{}.png'.format(i))
    plt.close()
