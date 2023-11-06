# Imports

# > Standard Library
import os
import sys
import random

# > Local dependencies
from vis_arg_parser import get_args

# Add the above directory to the path
sys.path.append('..')
from model import CERMetric, WERMetric, CTCLoss

# > Third party libraries
import tensorflow as tf
import numpy as np
import cv2

# Prep GPU support and set seeds/objects
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
    print('Please provide a path to an existing model directory with --existing_model')
    exit(1)

SEED = args.seed
GPU = args.gpu
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
if GPU >= 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

tf.keras.utils.get_custom_objects().update({"CERMetric": CERMetric})
tf.keras.utils.get_custom_objects().update({"WERMetric": WERMetric})
tf.keras.utils.get_custom_objects().update({"CTCLoss": CTCLoss})


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    model_channels = model.input_shape[3]
    print(model.summary())

    if args.sample_image:
        if not os.path.exists(args.sample_image):
            print('cannot find textline img on disk: ' + args.sample_image)
            exit(1)
        img_path = args.sample_image
    else:
        print(
            'Please provide a path to a text line img with--sample_image parameter')
        exit(1)

    # Remake data_generator parts
    target_height = 64
    original_image = tf.io.read_file(img_path)
    original_image = tf.image.decode_png(original_image, channels=model_channels)
    original_image = tf.image.resize(original_image,
                                     [target_height,
                                      tf.cast(target_height * tf.shape(original_image)[1]
                                              / tf.shape(original_image)[0], tf.int32)],
                                     preserve_aspect_ratio=True)

    image_width = tf.shape(original_image)[1]
    image_height = tf.shape(original_image)[0]
    original_image = tf.image.resize_with_pad(original_image, target_height, image_width + 50)

    # Normalize the image and something else
    img = 0.5 - (original_image / 255)
    img = tf.transpose(img, perm=[1, 0, 2])
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    preds = tf.dtypes.cast(preds, tf.float32)
    char_list = MODEL_PATH + "charlist.txt"
    with open(char_list, 'r') as f:
        char_list = f.read()

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    timestep_char_list_indices = []
    timestep_char_list_indices_top_5 = []
    top_k = 3
    step_width = 0
    pad_steps_skip = 0
    for text_line in preds:
        timesteps = len(text_line)
        step_width = tf.get_static_value(image_width + 50) / timesteps
        pad_steps_skip = np.floor(50 / step_width)
        for time_step in text_line:
            timestep_char_list_indices.append(tf.get_static_value(tf.math.argmax(time_step)))
            timestep_char_list_indices_top_5.append(tf.get_static_value(tf.math.top_k(time_step, k=top_k, sorted=True)))

    def remove_tags(text):
        text = text.replace('␃', '')  # public static String STRIKETHROUGHCHAR = "␃"; //Unicode Character “␃” (U+2403)
        text = text.replace('␅', '')  # public static String UNDERLINECHAR = "␅"; //Unicode Character “␅” (U+2405)
        text = text.replace('␄', '')  # public static String SUBSCRIPTCHAR = "␄"; // Unicode Character “␄” (U+2404)
        text = text.replace('␆', '')  # public static String SUPERSCRIPTCHAR = "␆"; // Unicode Character “␆” (U+2406)
        return text

    original_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    original_image_padded = cv2.resize(original_image, (tf.get_static_value(image_width + 50), 64))

    bordered_img = cv2.copyMakeBorder(original_image_padded, 50, 200, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.putText(bordered_img, "Time-step predictions (Top-1 prediction):",
                org=(0, 15),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=2)

    timestep_char_labels_cleaned = []
    line_start = 25

    # if blank token in char_list -> take normal else i-1
    if '' in char_list:
        index_correction = 0
    else:
        index_correction = -1

    # Retrieve the top 5 predictions for each timestep and write them down underneath each other
    for index, char_index in enumerate(timestep_char_list_indices):
        if index < pad_steps_skip:
            continue
        line_start += step_width
        start_point = (int(line_start), 50)
        end_point = (int(line_start), tf.get_static_value(image_height) + 50)
        if char_index < len(char_list) + 1:  # char_list + 1 is blank token, which is final character
            timestep_char_labels_cleaned.append(remove_tags(char_list[char_index + index_correction]))
            if char_list[char_index] == " ":  # Do not draw predictions that are spaces for better readability
                continue
            cv2.line(bordered_img,
                     start_point,
                     end_point,
                     color=(0, 0, 255),
                     thickness=1)
            cv2.putText(bordered_img,
                        remove_tags(char_list[char_index + index_correction]),
                        org=start_point,
                        color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        thickness=1)

            # Add the top-2 to top-5 most probable characters for the current time step
            table_start_height = 170
            for top_char_index in timestep_char_list_indices_top_5[index].indices.numpy()[1:]:
                if top_char_index < len(char_list) + 1:
                    cv2.putText(bordered_img,
                                remove_tags(char_list[top_char_index + index_correction]),
                                org=(int(line_start), table_start_height),
                                color=(0, 0, 255),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                thickness=1)
                    table_start_height += 50

    # Add the top-5 for the specific timestep
    row_num = 2
    min_row_y = 170
    max_row_y = 170 + ((top_k - 1) * 50)
    for row_height in range(170, max_row_y, 50):
        # Draw the top-5 text
        cv2.putText(bordered_img, "Top-" + str(row_num),
                    org=(0, row_height),
                    color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.5,
                    thickness=1)
        row_num += 1

    # Add misc. text
    cv2.putText(bordered_img, "Other predictions for this time step (lower probability):",
                org=(0, 140),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                thickness=1)
    cv2.putText(bordered_img, "Final result (collapsed blank characters):",
                org=(0, 270),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                thickness=1)

    # Add the cleaned version of the prediction
    cv2.putText(bordered_img,
                "".join(timestep_char_labels_cleaned),
                org=(50, 300),
                color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                thickness=1)
    print("".join(timestep_char_labels_cleaned))
    cv2.imwrite("output/timestep_prediction_plot.jpg", bordered_img)

if __name__ == "__main__":
    main()
