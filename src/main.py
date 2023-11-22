import logging
import os
import matplotlib.pyplot as plt
import subprocess
import uuid
import json
import re
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
    # public static String STRIKETHROUGHCHAR = "␃"; //Unicode Character “␃” (U+2403)
    text = text.replace('␃', '')
    # public static String UNDERLINECHAR = "␅"; //Unicode Character “␅” (U+2405)
    text = text.replace('␅', '')
    # public static String SUBSCRIPTCHAR = "␄"; // Unicode Character “␄” (U+2404)
    text = text.replace('␄', '')
    # public static String SUPERSCRIPTCHAR = "␆"; // Unicode Character “␆” (U+2406)
    text = text.replace('␆', '')
    return text


def preprocess_text(text):
    text = text.strip().replace('', '')
    text = remove_tags(text)
    return text


def print_predictions(filename, original_text, predicted_text, char_str=None):
    logging.info("--------------------------------------------------------")
    logging.info(f"File: {filename}")
    logging.info(f"Original text  - {original_text}")
    logging.info(f"Predicted text - {predicted_text}")
    if char_str:
        [logging.info(s) for s in char_str]


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


def setup_word_beam_search(args, charlist):
    # Check if the corpus file exists
    if not os.path.exists(args.corpus_file):
        raise FileNotFoundError(f'Corpus file not found: {args.corpus_file}')

    with open(args.corpus_file) as f:
        corpus = f.read()
    logging.info(f'Using corpus file: {args.corpus_file}')

    word_chars = \
        '-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzßàáâçèéëïñôöûüň'
    chars = '' + ''.join(sorted(list(charlist)))

    wbs = WordBeamSearch(args.beam_width, 'NGrams', args.wbs_smoothing,
                         corpus.encode('utf8'), chars.encode('utf8'),
                         word_chars.encode('utf8'))
    logging.info('Created WordBeamSearch')
    return wbs


def process_batch(batch, prediction_model, utilsObject,
                  args, wbs, loader, batch_no, chars):
    X, y_true = batch

    # Get the predictions
    predictions = prediction_model.predict(X, verbose=0)
    y_pred = decode_batch_predictions(
        predictions, utilsObject, args.greedy,
        args.beam_width, args.num_oov_indices)[0]

    # Transpose the predictions for WordBeamSearch
    predsbeam = tf.transpose(predictions, perm=[1, 0, 2])
    if wbs:
        # char_str = handle_wbs_results(predsbeam, wbs, args, chars)
        pass

    # Get the original texts
    orig_texts = [tf.strings.reduce_join(utilsObject.num_to_char(label))
                  .numpy().decode("utf-8").strip() for label in y_true]

    # Initialize counters
    n = 0
    batch_info = {
        'batch_edit_distance': 0,
        'batch_lower_edit_distance': 0,
        'batch_simple_edit_distance': 0,
        'batch_cer': 0,
        'batch_lower_cer': 0,
        'batch_simple_cer': 0,
        'batch_length': 0,
        'batch_length_simple': 0,
        'batch_norm_edit_distance': 0,
    }

    # Print the predictions and process the CER
    for (confidence, prediction), original_text in zip(y_pred, orig_texts):
        # Preprocess the text for CER calculation
        prediction = preprocess_text(prediction)
        original_text = preprocess_text(original_text)
        lower_prediction = prediction.lower()
        lower_original_text = original_text.lower()
        simple_prediction = re.sub(r'[^a-zA-Z0-9]', '', lower_prediction)
        simple_original_text = re.sub(r'[^a-zA-Z0-9]', '',
                                      original_text.lower())

        filename = loader.get_item(
            'validation', (batch_no * args.batch_size) + n)
        print_predictions(filename, original_text, prediction)

        # Calculate edit distance
        edit_distance = editdistance.eval(prediction, original_text)
        lower_edit_distance = editdistance.eval(lower_prediction,
                                                lower_original_text)
        simple_edit_distance = editdistance.eval(simple_prediction,
                                                 simple_original_text)

        # Calculate CER
        cer = edit_distance / max(len(original_text), 1)
        lower_cer = lower_edit_distance / max(len(lower_original_text), 1)
        simple_cer = simple_edit_distance / max(len(simple_original_text), 1)

        logging.info(f"Confidence = {confidence:.4f}")
        logging.info(
            f"CER        = {cer:.4f} ({edit_distance}/{len(original_text)})")
        logging.info(
            f"Lower CER  = {lower_cer:.4f} ({lower_edit_distance}/"
            f"{len(lower_original_text)})")
        logging.info(
            f"Simple CER = {simple_cer:.4f} ({simple_edit_distance}/"
            f"{len(simple_original_text)})")

        if wbs:
            pass

        if args.normalization_file:
            pass

        # Update the counters
        batch_info['batch_edit_distance'] += edit_distance
        batch_info['batch_lower_edit_distance'] += lower_edit_distance
        batch_info['batch_simple_edit_distance'] += simple_edit_distance
        batch_info['batch_length'] += len(original_text)
        batch_info['batch_length_simple'] += len(simple_original_text)

        n += 1

    # Normalize the CER
    batch_info['batch_cer'] = batch_info["batch_edit_distance"] \
        / max(batch_info["batch_length"], 1)
    batch_info['batch_lower_cer'] = batch_info["batch_lower_edit_distance"] \
        / max(batch_info["batch_length"], 1)
    batch_info['batch_simple_cer'] = batch_info["batch_simple_edit_distance"] \
        / max(batch_info["batch_length_simple"], 1)

    return batch_info


def display_statistics(batch_stats, total_stats, metrics):
    headers = ["Metric", "Batch", "Total"]
    row_format = "{:>15} | {:>15} | {:>15}"

    logging.info("Validation statistics:")
    logging.info("=" * 53)

    # Print header
    logging.info(row_format.format(*headers))
    logging.info("-" * 53)

    # Print each metric row
    for metric, batch_value, total_value in zip(metrics, batch_stats,
                                                total_stats):
        batch_value_str = f"{batch_value:.4f}" if isinstance(
            batch_value, float) else str(batch_value)
        total_value_str = f"{total_value:.4f}" if isinstance(
            total_value, float) else str(total_value)
        logging.info(row_format.format(
            metric, batch_value_str, total_value_str))

    logging.info("=" * 53)


def perform_validation(args, model, validation_dataset, char_list, dataloader):
    logging.info("Performing validation...")

    utils_object = Utils(char_list, args.use_mask)
    prediction_model = get_prediction_model(model)

    # Setup WordBeamSearch if needed
    wbs = setup_word_beam_search(args, char_list) if args.corpus_file else None

    # Initialize variables for CER calculation
    n_items = 0
    total_edit_distance, total_edit_distance_lower, total_edit_distance_simple\
        = 0, 0, 0
    total_length, total_length_simple = 0, 0
    norm_total_cer, norm_total_cer_lower = 0, 0
    norm_total_length = 0

    # Process each batch in the validation dataset
    for batch_no, batch in enumerate(validation_dataset):
        # Logic for processing each batch, calculating CER, etc.
        batch_info = process_batch(batch, prediction_model, utils_object, args,
                                   wbs, dataloader, batch_no, char_list)

        # Update the counters
        total_edit_distance += batch_info['batch_edit_distance']
        total_edit_distance_lower += batch_info['batch_lower_edit_distance']
        total_edit_distance_simple += batch_info['batch_simple_edit_distance']

        total_length += batch_info['batch_length']
        total_length_simple += batch_info['batch_length_simple']

        n_items += len(batch[1])

        # Calculate the new total CER
        total_cer = total_edit_distance / max(total_length, 1)
        total_cer_lower = total_edit_distance_lower / \
            max(total_length, 1)
        total_cer_simple = total_edit_distance_simple / \
            max(total_length_simple, 1)

        # Calculate the normalized CER
        if args.normalization_file:
            pass

        # Print batch info
        metrics = ['CER', 'Lower CER', 'Simple CER', 'Items']
        batch_stats = [batch_info['batch_cer'], batch_info['batch_lower_cer'],
                       batch_info['batch_simple_cer'], len(batch[1])]
        total_stats = [total_cer, total_cer_lower, total_cer_simple, n_items]

        display_statistics(batch_stats, total_stats, metrics)

    # TODO: Calculate and print final CER results


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

    # Train the model
    if args.do_train:
        history = train_model(model, args, training_dataset,
                              validation_dataset, loader)

        # Plot the training history
        plot_training_history(history, args)

    # Evaluate the model
    if args.do_validate:
        perform_validation(args, model, validation_dataset, charlist, loader)

    # Infer with the model
    if args.do_inference:
        pass


if __name__ == "__main__":
    main()
