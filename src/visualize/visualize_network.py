# Imports

# > Standard Library
import os
import random
import sys

# > Local dependencies
from vis_arg_parser import get_args

# Add the above directory to the path
sys.path.append('..')
from model import CERMetric, WERMetric, CTCLoss

# > Third party libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    # Compute gradients
    grads = tape.gradient(loss, img)
    # Normalize gradients
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def deprocess_image(img):
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def initialize_image(channels):
    if args.sample_image:
        target_height = 64
        img_path = args.sample_image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=model_channels)
        img = tf.image.resize(img,
                                     [target_height,
                                      tf.cast(target_height * tf.shape(img)[1]
                                              / tf.shape(img)[0], tf.int32)],
                                     preserve_aspect_ratio=True)
    else:
        # We start from a gray image with some random noise (64 height x 128 width)
        img = tf.random.uniform((1, 128, 64, channels))
    return (img - 0.5) * 0.25


def visualize_filter(filter_index, channels):
    # We run gradient ascent for 20 steps
    iterations = 20
    learning_rate = 10
    img = initialize_image(channels)
    for iteration in range(iterations):
        if (args.sample_image):
            img = np.expand_dims(img, axis=0)
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)
    img = tf.transpose(img[0].numpy(), perm=[1, 0, 2])
    # Decode the resulting input image
    img = deprocess_image(img)
    return loss, img


def select_number_of_row_plots():
    layer_info = []
    for layer in model.layers:
        layer_name = layer.name.lower()
        if layer_name.startswith("conv"):
            layer_info.append({"layer_name": layer_name,
                               "kernel_size": str(layer.kernel_size),
                               "filters": str(layer.filters)})

    if args.do_detailed:
        return layer_info, []
    else:
        # Create dictionaries to track the first occurrences of third element (i.e. filters).
        filter_occurrence_dict = {}
        sub_list_indices = []
        for i, layer_data in enumerate(layer_info):
            filter = layer_data.get("filters")
            if filter not in filter_occurrence_dict:
                filter_occurrence_dict[filter] = i
                sub_list_indices.append(i)
        # Retrieve only the elements from layer_info for the specific sub_list_indices
        return [layer_info[i] for i in sub_list_indices], sub_list_indices

args = get_args()
if args.existing_model:
    if not os.path.exists(args.existing_model):
        print('cannot find existing model on disk: ' + args.existing_model)
        exit(1)
    MODEL_PATH = args.existing_model
else:
    print('Please provide a path to an existing model directory with --existing_model')
    exit(1)

SEED = args.seed
GPU = args.gpu
base_output = args.output

if not os.path.exists(base_output):
    os.makedirs(base_output)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
if GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # if len(gpus) > 0:
    #     tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
    #         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# Option for plotting compact / full layers
# If compact then one row is plotted for each conv2d on the filter # level (so one for 16 filters, one for 32 etc.)
# Base is compact otherwise --plot_detailed

tf.keras.utils.get_custom_objects().update({"CERMetric": CERMetric})
tf.keras.utils.get_custom_objects().update({"WERMetric": WERMetric})
tf.keras.utils.get_custom_objects().update({"CTCLoss": CTCLoss})

model = tf.keras.models.load_model(MODEL_PATH)
model_channels = model.layers[0].input_shape[0][3]
print(model.summary())

# Prep plots
num_filters = 10  # Number of filters per row
layer_info, sub_list_indices = select_number_of_row_plots()
n_conv_layers = len(layer_info)  # Number of rows

#Top level plot
plt.style.use('dark_background')
fig = plt.figure(figsize=(2 * num_filters, 2 * n_conv_layers))
sub_figs = fig.subfigures(n_conv_layers, 1, wspace=0.1)

row_index = 0
conv_layer_list = []
for layer in model.layers:
    if layer.name.lower().startswith("conv"):
        conv_layer_list.append(layer)

if len(sub_list_indices) > 0:
    layer_list = [conv_layer_list[i] for i in sub_list_indices]
else:
    layer_list = conv_layer_list

for layer in layer_list:
    if not layer.name.lower().startswith("conv"):
        continue
    if row_index == n_conv_layers:
        break

    print("Plotting filters for layer: ", layer_info[row_index])
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # Sub_plot level
    filter_plot = sub_figs[row_index].subplots(1, num_filters)

    # Check the number of filters for this layer and take num_filters times a random number out of that number
    random_filter_indices = random.sample(range(int(layer_info[row_index].get("filters"))), num_filters)
    filter_images = []
    for filter_index in range(num_filters):
        try:
            loss, img = visualize_filter(random_filter_indices[filter_index], model_channels)
            filter_images.append(img)

            # Individual plot level
            filter_plot[filter_index].imshow(filter_images[filter_index], aspect='auto',cmap='gray')
        except IndexError:
            "filter_index has surpassed the filters in the layer, select a lower number of filters to plot"

    # Disable axis
    for ax in filter_plot:
        ax.axis("off")

    # Display the layer_name to the left of the subplots
    sub_figs[row_index].suptitle(layer_info[row_index])
    plt.savefig("layer_filter_plots.png", dpi=300)
    row_index += 1
