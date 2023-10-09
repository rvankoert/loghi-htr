# Imports

# > Standard Library
import random
import os
import sys

# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from data_loader import DataLoader
from model import CERMetric, WERMetric, CTCLoss
from utils import initialize_image, deprocess_image, get_feature_maps
from vis_arg_parser import get_args

# > Third party libraries
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import get_custom_objects

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

def main():
    model = keras.models.load_model(MODEL_PATH)
    model_channels = model.layers[0].input_shape[0][3]
    img_size = (args.height, args.width, model_channels)
    submodel = model
    print(submodel.summary())

    def compute_loss(input_image, filter_index):
        """
        Compute the loss for a given filter in a convolutional neural network layer.

        Parameters:
        - input_image (tf.Tensor): The input image tensor.
        - filter_index (int): Index of the filter in the convolutional layer.

        Returns:
        - tf.Tensor: The computed loss value as the mean activation of the specified filter.
        """
        activation = feature_extractor(input_image)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

    def gradient_ascent_step(img, filter_index, learning_rate):
        """
        Perform a single step of gradient ascent to visualize a convolutional filter.

        Parameters:
        - img (tf.Tensor): The input image tensor to be updated during gradient ascent.
        - filter_index (int): Index of the filter in the convolutional layer.
        - learning_rate (float): The learning rate for gradient ascent.

        Returns:
        - Tuple[tf.Tensor, tf.Tensor]: A tuple containing the computed loss value and
          the updated input image tensor after a single gradient ascent step.
        """
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
        """
        Visualize a convolutional filter using gradient ascent.

        Parameters:
        - filter_index (int): Index of the filter in the convolutional layer.
        - channels (int): Number of channels in the input image.

        Returns:
        - Tuple[tf.Tensor, np.ndarray]: A tuple containing the computed loss value and the
          resulting visualized image as a NumPy array.
        """
        # We run gradient ascent for a specified number of steps
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
        feature_extractor = keras.Model(
            inputs=submodel.inputs, outputs=layer.output)

        all_imgs = []
        numFilters = layer.output_shape[3]
        i = 0
        for filter_index in range(numFilters):
            print("Processing filter %d" % (filter_index,))
            loss, img = visualize_filter(filter_index, model_channels)

            all_imgs.append(img)

        char_list = None
        if args.validation_list:
            if not os.path.exists(args.validation_list):
                print('cannot find validation .txt on disk: ' + args.validation_list)
                exit(1)
        else:
            print('Please provide a path to a .txt containing (validation) images, e.g. the sample_list.txt inside "loghi-htr/tests/data" ')
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
                if model_channels == 1:
                    maps[j - 1] = tf.squeeze(maps[j - 1])
                plt.imshow(maps[j - 1]+0.5, cmap='gray')

            filename = loader.get_item(
                'inference', (batch_counter * args.batch_size))
            plt.tight_layout()
            #Make sure that a "results" directory exists
            plt.savefig('results/{}-{}'.format(layerId,
                        os.path.basename(filename)))
            plt.close()
            batch_counter = batch_counter + 1

if __name__ == "__main__":
    main()