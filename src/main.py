import logging
import os
import matplotlib.pyplot as plt
import subprocess
import uuid
import json
import time
import re
from collections import defaultdict
# Import other necessary libraries

from arg_parser import get_args
from utils import load_model_from_directory, Utils, decode_batch_predictions
from model import CERMetric, WERMetric, CTCLoss, replace_recurrent_layer, \
    replace_final_layer, train_batch
from custom_layers import ResidualBlock
from vgsl_model_generator import VGSLModelGenerator
from data_loader import DataLoader


import tensorflow as tf
from word_beam_search import WordBeamSearch
import editdistance


def setup_environment(args):
    # Initial setup
    logging.info(f"Running with args: {vars(args)}")

    # Set the GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    logging.info(f"Available GPUs: {gpu_devices}")

    # Set the active GPUs depending on the 'gpu' argument
    if args.gpu == "-1":
        active_gpus = []
        logging.info("Using CPU")
    else:
        gpus = args.gpu.split(',')
        active_gpus = [gpu if str(i) in gpus else None for i,
                       gpu in enumerate(gpu_devices)]
        logging.info(f"Using GPU(s): {active_gpus}")

    tf.config.set_visible_devices(active_gpus, 'GPU')

    # Initialize the strategy
    strategy = initialize_strategy(args.use_float32, args.gpu)

    return strategy


def setup_logging():
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Remove the default Tensorflow logger handlers and use our own
    tf_logger = logging.getLogger('tensorflow')
    while tf_logger.handlers:
        tf_logger.handlers.pop()


def initialize_strategy(use_float32, gpu):
    # Set the strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Set mixed precision policy
    if not use_float32 and gpu != "-1":
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Using mixed_float16 precision")
    else:
        logging.info("Using float32 precision")

    return strategy


def load_initial_charlist(charlist_location, existing_model,
                          output_directory, replace_final_layer):
    # Set the character list location
    if not charlist_location and existing_model:
        charlist_location = existing_model + '/charlist.txt'
    elif not charlist_location:
        charlist_location = output_directory + '/charlist.txt'

    # Load the character list
    char_list = []

    # We don't need to load the charlist if we are replacing the final layer
    if not replace_final_layer:
        if os.path.exists(charlist_location):
            with open(charlist_location) as file:
                char_list = [char for char in file.read()]
            logging.info(f"Using charlist from: {charlist_location}")
        else:
            raise FileNotFoundError(
                f"Charlist not found at: {charlist_location} and "
                "replace_final_layer is False. Exiting...")

        logging.info(f"Using charlist: {char_list}")
        logging.info(f"Charlist length: {len(char_list)}")

    return char_list


def adjust_model_for_float32(model):
    # Recreate the exact same model but with float32
    config = model.get_config()

    # Set the dtype policy for each layer in the configuration
    for layer_config in config['layers']:
        if 'dtype' in layer_config['config']:
            layer_config['config']['dtype'] = 'float32'
        if 'dtype_policy' in layer_config['config']:
            layer_config['config']['dtype_policy'] = {
                'class_name': 'Policy',
                'config': {'name': 'float32'}}

    # Create a new model from the modified configuration
    model_new = tf.keras.Model.from_config(config)
    model_new.set_weights(model.get_weights())

    model = model_new

    # Verify float32
    for layer in model.layers:
        assert layer.dtype_policy.name == 'float32'

    return model


def initialize_data_loader(args, char_list, model):
    model_height = model.layers[0].input_shape[0][2]
    model_channels = model.layers[0].input_shape[0][3]
    img_size = (model_height, args.width, model_channels)

    return DataLoader(
        batch_size=args.batch_size,
        img_size=img_size,
        train_list=args.train_list,
        validation_list=args.validation_list,
        test_list=args.test_list,
        inference_list=args.inference_list,
        char_list=char_list,
        do_binarize_sauvola=args.do_binarize_sauvola,
        do_binarize_otsu=args.do_binarize_otsu,
        multiply=args.multiply,
        augment=args.augment,
        elastic_transform=args.elastic_transform,
        random_crop=args.random_crop,
        random_width=args.random_width,
        check_missing_files=args.check_missing_files,
        distort_jpeg=args.distort_jpeg,
        replace_final_layer=args.replace_final_layer,
        normalization_file=args.normalization_file,
        use_mask=args.use_mask,
        do_random_shear=args.do_random_shear
    )


def customize_model(model, args, charlist):
    # Replace certain layers if specified
    if args.replace_recurrent_layer:
        logging.info("Replacing recurrent layer with "
                     f"{args.replace_recurrent_layer}")
        model = replace_recurrent_layer(model,
                                        len(charlist),
                                        args.replace_recurrent_layer,
                                        use_mask=args.use_mask)

    # Replace the final layer if specified
    if args.replace_final_layer:
        new_classes = len(charlist) + 2 if args.use_mask else len(charlist) + 1
        logging.info(f"Replacing final layer with {new_classes} classes")
        model = replace_final_layer(model, len(
            charlist), model.name, use_mask=args.use_mask)

    # Freeze or thaw layers if specified
    if any([args.thaw, args.freeze_conv_layers,
            args.freeze_recurrent_layers, args.freeze_dense_layers]):
        for layer in model.layers:
            if args.thaw:
                layer.trainable = True
                logging.info(f"Thawing layer: {layer.name}")
            elif args.freeze_conv_layers and \
                (layer.name.lower().startswith("conv") or
                 layer.name.lower().startswith("residual")):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False
            elif args.freeze_recurrent_layers and \
                    layer.name.lower().startswith("bidirectional"):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False
            elif args.freeze_dense_layers and \
                    layer.name.lower().startswith("dense"):
                logging.info(f"Freezing layer: {layer.name}")
                layer.trainable = False

    # Further configuration based on use_float32 and gpu
    if args.use_float32 or args.gpu == "-1":
        # Adjust the model for float32
        logging.info("Adjusting model for float32")
        model = adjust_model_for_float32(model)

    return model


def load_or_create_model(args, custom_objects, char_list):
    if args.existing_model:
        model = load_model_from_directory(
            args.existing_model, custom_objects=custom_objects)
        if args.model_name:
            model._name = args.model_name
    else:
        model_generator = VGSLModelGenerator(
            model=args.model,
            name=args.model_name,
            output_classes=len(char_list) + 2
            if args.use_mask else len(char_list) + 1
        )
        model = model_generator.build()

    return model


def get_optimizer(optimizer_name, learning_rate_schedule):
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "adamw": tf.keras.optimizers.experimental.AdamW,
        "adadelta": tf.keras.optimizers.Adadelta,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adamax": tf.keras.optimizers.Adamax,
        "adafactor": tf.keras.optimizers.Adafactor,
        "nadam": tf.keras.optimizers.Nadam
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name](learning_rate=learning_rate_schedule)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")


def create_learning_rate_schedule(learning_rate, decay_rate, decay_steps,
                                  train_batches, do_train):
    if decay_rate > 0 and do_train:
        if decay_steps > 0:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate
            )
        elif decay_steps == -1:
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=train_batches,
                decay_rate=decay_rate
            )
    return learning_rate


def verify_charlist_length(charlist, model, use_mask):
    # Verify that the length of the charlist is correct
    if use_mask:
        expected_length = model.layers[-1].output_shape[2] - 2
    else:
        expected_length = model.layers[-1].output_shape[2] - 1
    if len(charlist) != expected_length:
        raise ValueError(
            f"Charlist length ({len(charlist)}) does not match "
            f"model output length ({expected_length}). If the charlist "
            "is correct, try setting use_mask to True.")


def save_charlist(charlist, output, output_charlist_location=None):
    # Save the new charlist
    if not output_charlist_location:
        output_charlist_location = output + '/charlist.txt'
    with open(output_charlist_location, 'w') as chars_file:
        chars_file.write(str().join(charlist))

############## Metadata functions ##############


def get_git_hash():
    if os.path.exists("version_info"):
        with open("version_info") as file:
            return file.read().strip()
    else:
        try:
            result = subprocess.run(['git', 'log', '--format=%H', '-n', '1'],
                                    stdout=subprocess.PIPE,
                                    check=True)
            return result.stdout.decode('utf-8').strip().replace('"', '')
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess failed: {e}")
        except OSError as e:
            logging.error(f"OS error occurred: {e}")
        return "Unavailable"


def summarize_model(model):
    model_layers = []
    model.summary(print_fn=lambda x: model_layers.append(x))
    return model_layers


def get_config(args, model):
    return {
        'git_hash': get_git_hash(),
        'args': vars(args),
        'model': summarize_model(model),
        'notes': ' ',
        'uuid': str(uuid.uuid4())
    }


def store_info(args, model):
    config = get_config(args, model)
    config_file_output = args.config_file_output if args.config_file_output \
        else os.path.join(args.output, 'config.json')

    try:
        with open(config_file_output, 'w') as configuration_file:
            json.dump(config, configuration_file)
    except IOError as e:
        logging.error(f"Error writing config file: {e}")


############## Training functions ##############

def train_model(model, args, training_dataset, validation_dataset, loader):
    metadata = get_config(args, model)

    history = train_batch(
        model,
        training_dataset,
        validation_dataset,
        epochs=args.epochs,
        output=args.output,
        model_name=model.name,
        steps_per_epoch=args.steps_per_epoch,
        max_queue_size=args.max_queue_size,
        early_stopping_patience=args.early_stopping_patience,
        output_checkpoints=args.output_checkpoints,
        charlist=loader.charList,
        metadata=metadata,
        verbosity_mode=args.training_verbosity_mode
    )

    return history


def plot_training_history(history, args):
    def plot_metric(metric, title, filename):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history.history[metric], label=metric)
        if args.validation_list:
            plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/CER")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output, filename))

    plot_metric("loss", "Training Loss", 'loss_plot.png')
    plot_metric("CER_metric", "Character Error Rate (CER)", 'cer_plot.png')


############## Validation util functions ##############

def remove_tags(text):
    return re.sub(r'[␃␅␄␆]', '', text)


def preprocess_text(text):
    text = text.strip().replace('', '')
    text = remove_tags(text)
    return text


def print_predictions(filename, original_text, predicted_text, char_str=None):
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info(f"File: {filename}")
    logging.info(f"Original text  - {original_text}")
    logging.info(f"Predicted text - {predicted_text}")
    if char_str:
        logging.info(f"WordBeamSearch - {char_str}")
    logging.info("")


def simplify_text(text):
    lower_text = text.lower()
    simple_text = re.sub(r'[^a-zA-Z0-9]', '', lower_text)
    return lower_text, simple_text


def display_statistics(batch_stats, total_stats, metrics):
    # Find the maximum length of metric names
    max_metric_length = max(len(metric) for metric in metrics)

    # Prepare headers and format strings
    headers = ["Metric", "Batch", "Total"]
    header_format = "{:>" + str(max_metric_length) + "} | {:>7} | {:>7}"
    row_format = "{:>" + str(max_metric_length) + "} | {:>7} | {:>7}"
    separator = "-" * (max_metric_length + 21)
    border = "=" * (max_metric_length + 21)

    logging.info("Validation metrics:")
    logging.info(border)

    # Print header
    logging.info(header_format.format(*headers))
    logging.info(separator)

    # Print each metric row
    for metric, batch_value, total_value in zip(metrics, batch_stats,
                                                total_stats):
        batch_value_str = f"{batch_value:.4f}" if isinstance(
            batch_value, float) else str(batch_value)
        total_value_str = f"{total_value:.4f}" if isinstance(
            total_value, float) else str(total_value)
        logging.info(row_format.format(
            metric, batch_value_str, total_value_str))

    logging.info(border)


def calculate_edit_distances(prediction, original_text):
    # Preprocess the text
    lower_prediction, simple_prediction = simplify_text(prediction)
    lower_original, simple_original = simplify_text(original_text)

    # Calculate edit distance
    edit_distance = editdistance.eval(prediction, original_text)
    lower_edit_distance = editdistance.eval(lower_prediction,
                                            lower_original)
    simple_edit_distance = editdistance.eval(simple_prediction,
                                             simple_original)

    return edit_distance, lower_edit_distance, simple_edit_distance


def edit_distance_to_cer(edit_distance, length):
    return edit_distance / max(length, 1)


def calculate_cers(info, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix

    edit_distance = info[prefix + 'edit_distance']
    length = info[prefix + 'length']
    lower_edit_distance = info[prefix + 'lower_edit_distance']
    length_simple = info[prefix + 'length_simple']
    simple_edit_distance = info[prefix + 'simple_edit_distance']

    # Calculate CER
    cer = edit_distance_to_cer(edit_distance, length)
    lower_cer = edit_distance_to_cer(lower_edit_distance, length)
    simple_cer = edit_distance_to_cer(simple_edit_distance, length_simple)

    return cer, lower_cer, simple_cer


def print_cer_stats(distances, lengths, prefix=""):
    prefix = f"{prefix} " if prefix else prefix

    edit_distance, lower_edit_distance, simple_edit_distance = distances
    length, length_simple = lengths

    # Calculate CER
    cer = edit_distance_to_cer(edit_distance, length)
    lower_cer = edit_distance_to_cer(lower_edit_distance, length)
    simple_cer = edit_distance_to_cer(simple_edit_distance, length_simple)

    # Print CER stats
    logging.info(f"{prefix}CER        = {cer:.4f} ({edit_distance}/{length})")
    logging.info(f"{prefix}Lower CER  = {lower_cer:.4f} ({lower_edit_distance}"
                 f"/{length})")
    logging.info(f"{prefix}Simple CER = {simple_cer:.4f} "
                 f"({simple_edit_distance}/{length_simple})")
    logging.info("")


def update_totals(info, total, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix

    edit_distance = info[prefix + 'edit_distance']
    length = info[prefix + 'length']
    lower_edit_distance = info[prefix + 'lower_edit_distance']
    length_simple = info[prefix + 'length_simple']
    simple_edit_distance = info[prefix + 'simple_edit_distance']

    total[prefix + 'edit_distance'] += edit_distance
    total[prefix + 'length'] += length
    total[prefix + 'lower_edit_distance'] += lower_edit_distance
    total[prefix + 'length_simple'] += length_simple
    total[prefix + 'simple_edit_distance'] += simple_edit_distance

    return total


def update_batch_info(info, distances, lengths, prefix=""):
    prefix = f"{prefix}_" if prefix else prefix
    edit_distance, lower_edit_distance, simple_edit_distance = distances
    length, length_simple = lengths

    info[f'{prefix}edit_distance'] += edit_distance
    info[f'{prefix}length'] += length
    info[f'{prefix}lower_edit_distance'] += lower_edit_distance
    info[f'{prefix}length_simple'] += length_simple
    info[f'{prefix}simple_edit_distance'] += simple_edit_distance

    return info


def process_cer_type(batch_info, total_counter, metrics, batch_stats,
                     total_stats, prefix=""):
    # Update totals
    updated_totals = update_totals(batch_info, total_counter, prefix=prefix)

    # Calculate CERs for both batch and total
    batch_cers = calculate_cers(batch_info, prefix=prefix)
    total_cers = calculate_cers(updated_totals, prefix=prefix)

    # Define metric names based on the prefix
    prefix = f"{prefix} " if prefix else prefix
    cer_names = [f"{prefix}CER", f"{prefix}Lower CER", f"{prefix}Simple CER"] \
        if prefix else ["CER", "Lower CER", "Simple CER"]

    # Extend metrics and stats
    metrics.extend(cer_names)
    batch_stats.extend(batch_cers)
    total_stats.extend(total_cers)

    return updated_totals, metrics, batch_stats, total_stats


def process_prediction_type(prediction, original, batch_info, prefix=""):
    # Preprocess the text for CER calculation
    _, simple_original = simplify_text(original)

    # Calculate edit distances
    distances = calculate_edit_distances(prediction, original)

    # Unpack the distances
    edit_distance, lower_edit_distance, simple_edit_distance = distances
    lengths = [len(original), len(simple_original)]

    # Print the predictions if there are any errors
    if edit_distance > 0:
        print_cer_stats(distances, lengths, prefix=prefix)

    # Update the counters
    batch_info = update_batch_info(batch_info,
                                   distances,
                                   lengths,
                                   prefix=prefix)

    return batch_info


def calculate_confidence_intervals(cer_metrics, n):
    intervals = []

    # Calculate the confidence intervals
    for cer_metric in cer_metrics:
        intervals.append(calc_95_confidence_interval(cer_metric, n))

    return intervals


############## Validation functions ##############


def get_prediction_model(model):
    last_dense_layer = None
    for layer in reversed(model.layers):
        if layer.name.startswith('dense'):
            last_dense_layer = layer
            break
    if last_dense_layer is None:
        raise ValueError("No dense layer found in the model")

    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, last_dense_layer.output
    )
    return prediction_model


def setup_word_beam_search(args, charlist, loader):
    logging.info("Setting up WordBeamSearch...")

    # Check if the corpus file exists
    if not os.path.exists(args.corpus_file):
        raise FileNotFoundError(f'Corpus file not found: {args.corpus_file}')

    # Load the corpus
    with open(args.corpus_file) as f:
        # Create the corpus
        corpus = ''
        for line in f:
            if args.normalization_file:
                line = loader.normalize(line, args.normalization_file)
            corpus += line
    logging.info(f'Using corpus file: {args.corpus_file}')

    # Create the WordBeamSearch object
    word_chars = \
        '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzßàáâçèéëïñôöûüň'
    chars = '' + ''.join(sorted(charlist))
    wbs = WordBeamSearch(args.beam_width, 'NGrams', args.wbs_smoothing,
                         corpus.encode('utf8'), chars.encode('utf8'),
                         word_chars.encode('utf8'))

    logging.info('Created WordBeamSearch')

    return wbs


def handle_wbs_results(predsbeam, wbs, args, chars):
    label_str = wbs.compute(predsbeam)
    char_str = []  # decoded texts for batch
    for curr_label_str in label_str:
        s = ''.join([chars[label-1] for label in curr_label_str])
        s = preprocess_text(s)
        char_str.append(s)

    return char_str


def calc_95_confidence_interval(cer_metric, n):
    """ Calculates the binomial confidence radius of the given metric
    based on the num of samples (n) and a 95% certainty number
    E.g. cer_metric = 0.10, certainty = 95 and n= 5500 samples -->
    conf_radius = 1.96 * ((0.1*(1-0.1))/5500)) ** 0.5 = 0.008315576
    This means with 95% certainty we can say that the True CER of the model is
    between 0.0917 and 0.1083 (4-dec rounded)
    """
    return 1.96 * ((cer_metric*(1-cer_metric))/n) ** 0.5


def process_batch(batch, prediction_model, utilsObject,
                  args, wbs, loader, batch_no, chars):
    X, y_true = batch

    # Get the predictions
    predictions = prediction_model.predict(X, verbose=0)
    y_pred = decode_batch_predictions(
        predictions, utilsObject, args.greedy,
        args.beam_width, args.num_oov_indices)[0]

    # Transpose the predictions for WordBeamSearch
    if wbs:
        predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
        char_str = handle_wbs_results(predsbeam, wbs, args, chars)
    else:
        char_str = None

    # Get the original texts
    orig_texts = [tf.strings.reduce_join(utilsObject.num_to_char(label))
                  .numpy().decode("utf-8").strip() for label in y_true]

    # Initialize the batch info
    batch_info = defaultdict(int)

    # Print the predictions and process the CER
    for index, (confidence, prediction) in enumerate(y_pred):
        # Preprocess the text for CER calculation
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(orig_texts[index])

        # Calculate edit distances here so we can use them for printing the
        # predictions
        distances = \
            calculate_edit_distances(prediction, original_text)

        # Print the predictions if there are any errors
        if distances[0] > 0:
            filename = loader.get_item('validation',
                                       (batch_no * args.batch_size) + index)
            wbs_str = char_str[index] if wbs else None
            print_predictions(filename, original_text,
                              prediction, wbs_str)
            logging.info(f"Confidence = {confidence:.4f}")

        batch_info = process_prediction_type(prediction,
                                             original_text,
                                             batch_info)

        if args.normalization_file:
            # Normalize the text
            normalized_prediction = loader.normalize(prediction,
                                                     args.normalization_file)
            normalized_original = loader.normalize(original_text,
                                                   args.normalization_file)

            # Process the normalized CER
            batch_info = process_prediction_type(normalized_prediction,
                                                 normalized_original,
                                                 batch_info,
                                                 prefix="Normalized")

        if wbs:
            # Process the WBS CER
            batch_info = process_prediction_type(char_str[index],
                                                 original_text,
                                                 batch_info,
                                                 prefix="WBS")

    return batch_info


def perform_validation(args, model, validation_dataset, char_list, dataloader):
    logging.info("Performing validation...")

    utils_object = Utils(char_list, args.use_mask)
    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(args, char_list, dataloader) \
        if args.corpus_file else None

    # Initialize variables for CER calculation
    n_items = 0
    total_counter = defaultdict(int)
    total_normalized_counter = defaultdict(int)
    total_wbs_counter = defaultdict(int)

    # Process each batch in the validation dataset
    for batch_no, batch in enumerate(validation_dataset):
        # Logic for processing each batch, calculating CER, etc.
        batch_info = process_batch(batch, prediction_model, utils_object, args,
                                   wbs, dataloader, batch_no, char_list)
        metrics, batch_stats, total_stats = [], [], []

        # Calculate the CER
        total_counter, metrics, batch_stats, total_stats \
            = process_cer_type(
                batch_info, total_counter, metrics, batch_stats, total_stats)

        # Calculate the normalized CER
        if args.normalization_file:
            total_normalized_counter, metrics, batch_stats, total_stats \
                = process_cer_type(
                    batch_info, total_normalized_counter, metrics, batch_stats,
                    total_stats, prefix="Normalized")

        # Calculate the WBS CER
        if wbs:
            total_wbs_counter, metrics, batch_stats, total_stats \
                = process_cer_type(
                    batch_info, total_wbs_counter, metrics, batch_stats,
                    total_stats, prefix="WBS")

        # Print batch info
        n_items += len(batch[1])
        metrics.append('Items')
        batch_stats.append(len(batch[1]))
        total_stats.append(n_items)

        display_statistics(batch_stats, total_stats, metrics)
        logging.info("")

    # Print the final validation statistics
    logging.info("--------------------------------------------------------")
    logging.info("")
    logging.info("Final validation statistics")
    logging.info("---------------------------")
    logging.info("")

    # Calculate the CER confidence intervals on all metrics except Items
    intervals = calculate_confidence_intervals(total_stats[:-1], n_items)

    # Print the final statistics
    for metric, total_value, interval in zip(metrics[:-1], total_stats[:-1],
                                             intervals):
        logging.info(f"{metric} = {total_value:.4f} +/- {interval:.4f}")

    logging.info(f"Items = {total_stats[-1]}")
    logging.info("")

##############  Main function  ##############


def main():
    setup_logging()

    # Get the arguments
    args = get_args()

    # Set up the environment
    strategy = setup_environment(args)

    # Create the output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Get the initial character list
    charlist = load_initial_charlist(
        args.charlist, args.existing_model,
        args.output, args.replace_final_layer)

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(
            args, custom_objects, charlist)

        # Initialize the Dataloader
        loader = initialize_data_loader(args, charlist, model)
        training_dataset, validation_dataset, test_dataset, \
            inference_generator, utilsObject, train_batches\
            = loader.generators()

        # Replace the charlist with the one from the data loader
        charlist = loader.charList

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, args, charlist)

        # Save the charlist
        verify_charlist_length(charlist, model, args.use_mask)
        save_charlist(charlist, args.output, args.output_charlist)

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            args.learning_rate, args.decay_rate, args.decay_steps,
            train_batches, args.do_train)

        # Create the optimizer
        optimizer = get_optimizer(args.optimizer, lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer, loss=CTCLoss, metrics=[CERMetric(
            greedy=args.greedy, beam_width=args.beam_width), WERMetric()])

    # Print the model summary
    model.summary()

    # Store the model info (i.e., git hash, args, model summary, etc.)
    store_info(args, model)

    # Store timestamps
    timestamps = {
        'start_time': time.time(),
    }

    # Train the model
    if args.do_train:
        tick = time.time()

        history = train_model(model, args, training_dataset,
                              validation_dataset, loader)

        # Plot the training history
        plot_training_history(history, args)

        timestamps['train_time'] = time.time() - tick

    # Evaluate the model
    if args.do_validate:
        logging.warning("Validation results are without special markdown tags")

        tick = time.time()
        perform_validation(args, model, validation_dataset, charlist, loader)

        timestamps['validate_time'] = time.time() - tick

    # Infer with the model
    if args.do_inference:
        pass

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    if args.do_train:
        logging.info(f"Training completed in {timestamps['train_time']:.2f} "
                     "seconds")
    if args.do_validate:
        logging.info("Validation completed in "
                     f"{timestamps['validate_time']:.2f} seconds")
    logging.info(f"Total time: {time.time() - timestamps['start_time']:.2f} "
                 "seconds")


if __name__ == "__main__":
    main()
