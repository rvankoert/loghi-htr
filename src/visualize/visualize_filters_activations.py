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


def visualize_filter(filter_index, channels, feature_extractor, image_width=256):
    iterations = 20
    learning_rate = 10
    img = tf.random.uniform((1, image_width, 64, channels))  # Initialize image
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(img)
            activation = feature_extractor(img)
            filter_activation = activation[:, 2:-2, 2:-2, filter_index]
            loss = tf.reduce_mean(filter_activation)
        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
    img = tf.transpose(img[0].numpy(), perm=[1, 0, 2])
    img = deprocess_image(img)
    return loss, img

def deprocess_image(img):
    img /= 2.0
    img += 0.5
    img *= 255.
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def select_number_of_row_plots(model):
    layer_info = [{"layer_name": layer.name.lower(),
                   "kernel_size": str(layer.kernel_size),
                   "filters": str(layer.filters)}
                  for layer in model.layers if layer.name.lower().startswith("conv")]

    if args.do_detailed:
        return layer_info, []
    else:
        filter_occurrence_dict = {}
        sub_list_indices = []
        for i, layer_data in enumerate(layer_info):
            filter_value = layer_data.get("filters")
            if filter_value not in filter_occurrence_dict:
                filter_occurrence_dict[filter_value] = i
                sub_list_indices.append(i)

        return [layer_info[i] for i in sub_list_indices], sub_list_indices

# Retrieve args and load model
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

#Set seed for plots to check changes in preprocessing
np.random.seed(SEED)
tf.random.set_seed(SEED)
if GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')

tf.keras.utils.get_custom_objects().update({"CERMetric": CERMetric})
tf.keras.utils.get_custom_objects().update({"WERMetric": WERMetric})
tf.keras.utils.get_custom_objects().update({"CTCLoss": CTCLoss})

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    model_channels = model.layers[0].input_shape[0][3]
    print(model.summary())

    # Prep plots
    num_filters_per_row = args.num_filters_per_row  # Number of filters per row
    layer_info, sub_list_indices = select_number_of_row_plots(model)

    # Top level plot
    if not args.light_mode:
        plt.style.use('dark_background')
    fig = plt.figure(figsize=(5 * num_filters_per_row, num_filters_per_row), dpi=200)
    sub_figs = fig.subfigures(len(layer_info), 1)

    # Get layout parameters for later plots
    fig.tight_layout()
    dic = {par: getattr(fig.subplotpars, par) for par in ["left", "right", "bottom", "wspace"]}

    # Set base row_index
    row_index = 0

    # Collect convolutional layers
    conv_layer_list = [layer for layer in model.layers if layer.name.lower().startswith("conv")]

    # Select relevant layers
    layer_list = conv_layer_list if len(sub_list_indices) == 0 else [conv_layer_list[i] for i in sub_list_indices]

    if len(sub_list_indices) > 0:
        layer_list = [conv_layer_list[i] for i in sub_list_indices]
    else:
        layer_list = conv_layer_list

    for layer in layer_list:
        if not layer.name.lower().startswith("conv"):
            continue

        print("Plotting filters for layer: ", layer_info[row_index])
        feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

        # Set subplot layout
        num_subplot_rows = 2 if args.sample_image else 1
        filter_plot = sub_figs[row_index].subplots(num_subplot_rows, num_filters_per_row)
        filter_cmap = 'viridis' if model_channels > 1 else 'gray'

        # Randomly select filter indices
        random_filter_indices = random.sample(range(int(layer_info[row_index].get("filters"))), num_filters_per_row)

        filter_images = []
        feature_maps = []
        for filter_index in range(num_filters_per_row):
            try:
                if args.sample_image:
                    # Prep input text line for prediction
                    target_height = 64
                    img_path = args.sample_image
                    img = tf.io.read_file(img_path)
                    img = tf.image.decode_png(img, channels=model_channels)
                    img = tf.image.resize(img,
                                          [target_height,
                                           tf.cast(target_height * tf.shape(img)[1]
                                                   / tf.shape(img)[0], tf.int32)],
                                          preserve_aspect_ratio=True)
                    image_width = tf.shape(img)[1]
                    img = 0.5 - (img / 255)
                    img = tf.transpose(img, perm=[1, 0, 2])
                    img = np.expand_dims(img, axis=0)
                    maps = feature_extractor.predict(img)

                    # Add the filter images
                    loss, img = visualize_filter(random_filter_indices[filter_index], model_channels, feature_extractor, image_width)
                    filter_images.append(img)

                    # Add the feature maps
                    feature_maps.append(maps[0, :, :, random_filter_indices[filter_index]].T)

                    # If filter_plot has 2 rows then fill each row else do the regular one
                    filter_plot[0, filter_index].imshow(filter_images[filter_index], cmap=filter_cmap)
                    filter_plot[0, filter_index].set_title("Conv Filter: " + str(filter_index))
                    filter_plot[1, filter_index].imshow(feature_maps[filter_index], cmap='viridis')
                    filter_plot[1, filter_index].set_title("Layer Activations: " + str(filter_index))

                else:
                    # Add the filter images
                    loss, img = visualize_filter(random_filter_indices[filter_index], feature_extractor, model_channels)
                    filter_images.append(img)

                    # Individual plot level
                    filter_plot[filter_index].imshow(filter_images[filter_index], cmap= filter_cmap)
                    filter_plot[filter_index].set_title("Conv Filter: " + str(filter_index))
            except IndexError:
                "filter_index has surpassed the filters in the layer, select a lower number of filters to plot"

        # Fix layout parameters and keep some extra space at the top for suptitle
        fig.subplots_adjust(**dic, top=0.7, hspace=0.5)

        # Disable axes
        for ax in filter_plot.ravel():
            ax.set_axis_off()

        # Display the layer_name above the subplots
        layer_info_dict = layer_info[row_index]
        sub_figs[row_index].suptitle(layer_info_dict.get("layer_name")
                                     + ": kernel_size: " + str(layer_info_dict.get("kernel_size"))
                                     + " : filters: " + str(layer_info_dict.get("filters")),
                                     fontsize=20)
        row_index += 1

        # Prepare for saving image
        output_dir = args.output

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_name = (model.name
                       + ("_1channel" if model_channels == 1 else "_" + str(model_channels) + "channels")
                       + ("_filters_act" if args.sample_image else "_filters")
                       + ("_light" if args.light_mode else "_dark")
                       + ("_detailed.png" if args.do_detailed else".png"))

        plt.savefig(os.path.join(output_dir, output_name), bbox_inches='tight')
if __name__ == '__main__':
    main()