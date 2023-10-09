# Imports

# > Standard Library
import os
import sys
import random

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from data_loader import DataLoader
from utils import initialize_image, deprocess_image
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
        'Please provide a path to a .txt containing (validation) images, e.g. the sample_list.txt inside "loghi-htr/tests/data" ')
    exit(1)

sample_image = initialize_image(model_channels)
# img = deprocess_image(sample_image)
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img = img.reshape((args.height, args.width, args.channels))
with open(IMG_PATH.replace(".png",".txt"),'r') as f:
    img_label = f.read()

# @tf.function
def compute_saliency(image, label):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

    gradients = tape.gradient(loss, image)
    return gradients

print([img_label])
saliency = compute_saliency(tf.convert_to_tensor(img), tf.convert_to_tensor([img_label]))
saliency = tf.reduce_max(tf.abs(saliency), axis=-1)  # Compute max absolute gradient
saliency /= saliency.numpy().max()  # Normalize to [0, 1]

plt.figure(figsize=(8, 8))
plt.imshow(sample_image[0], cmap='gray')
plt.imshow(saliency[0], cmap='jet', alpha=0.6)
plt.colorbar()
plt.title('Saliency Map')
plt.axis('off')
plt.show()
