# Imports

# > Standard Library
import random
import sys
import os

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from config import *
from vis_arg_parser import get_args
from data_loader import DataLoader
from model import CERMetric, WERMetric, CTCLoss


# > Third party libraries
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tensorflow.keras.utils import get_custom_objects

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

args = get_args()
SEED = args.seed
GPU = args.gpu

if args.existing_model:
    if not os.path.exists(args.existing_model):
        print('cannot find existing model on disk: ' + args.existing_model)
        exit(1)
    MODEL_PATH = args.existing_model

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
submodel = model.get_layer(index=2)

# Load images
# img2 = load_img('images/bear.jpg')

char_list = None
if args.validation_list:
    if not os.path.exists(args.validation_list):
        print('cannot find validation .txt on disk: ' + args.validation_list)
        exit(1)
else:
    print(
        'Please provide a path to a .txt containing (validation) images, e.g. the sample_list.txt inside "loghi-htr/tests/data" ')
    exit(1)

loader = DataLoader(args.batch_size, img_size,
                    train_list=None,
                    validation_list=None,
                    test_list=None,
                    inference_list=args.validation_list,
                    char_list=char_list,
                    check_missing_files=False
                    )

training_generator, validation_generator, test_generator, inference_generator, utils, train_batches = loader.generators()

def loss(output):
    # 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
    return output[0][0]


def model_modifier(model):
    model.layers[-1].activation = tf.keras.activations.linear
    return model


# while i < 100:
batch_counter = 0
i = 0
for batch in inference_generator:
    if batch_counter > 10:
        print('breaking')
        break
    item = batch[0]
    i = i + 1

    X = item

    # item = inference_generator.__getitem__(i)

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