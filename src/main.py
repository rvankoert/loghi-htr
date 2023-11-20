import logging
import os
# Import other necessary libraries

from arg_parser import get_args
from utils import load_model_from_directory
from model import CERMetric, WERMetric, CTCLoss, replace_recurrent_layer, \
    replace_final_layer
from custom_layers import ResidualBlock
from vgsl_model_generator import VGSLModelGenerator
from data_loader import DataLoader

import tensorflow as tf


def setup_environment(args):
    # Initial setup
    setup_logging()
    logging.info(f"Running with args : {vars(args)}")

    logging.info(f"Using GPU: {args.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    # Height adjustment logic
    if args.height != model_height:
        if args.no_auto:
            raise ValueError(
                f"Input height ({args.height}) differs from model height "
                f"({model_height}). Exiting because no_auto is set.")
        else:
            logging.info(
                f"Adjusting input height from {args.height} to {model_height}")
            args.height = model_height

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
    if args.use_float32 or args.gpu == -1:
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
            channels=args.channels,
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
        raise ValueError(f"Invalid optimizer name: {optimizer.name}")


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

##############  Main function  ##############


if __name__ == "__main__":
    # Get the arguments
    args = get_args()

    # Set up the environment
    strategy = setup_environment(args)

    # Create the output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Get the character list
    charlist = load_initial_charlist(
        args.charlist, args.existing_model,
        args.output, args.replace_final_layer)

    # Load the model
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock}
    with strategy.scope():
        model = load_or_create_model(
            args, custom_objects, charlist)

        # Initialize the Dataloader
        loader = initialize_data_loader(args, charlist, model)
        training_generator, validation_generator, test_generator, \
            inference_generator, utilsObject, train_batches \
            = loader.generators()

        # Replace the charlist
        charlist = loader.charList

        # Additional model customization
        model = customize_model(model, args, charlist)

        # Save the new charlist
        output_charlist_location = args.output_charlist
        if not output_charlist_location:
            output_charlist_location = args.output + '/charlist.txt'
        with open(output_charlist_location, 'w') as chars_file:
            chars_file.write(str().join(charlist))

        # Compile the model by creating the optimizer and loss function
        lr_schedule = create_learning_rate_schedule(
            args.learning_rate, args.decay_rate, args.decay_steps,
            train_batches, args.do_train)
        optimizer = get_optimizer(args.optimizer, lr_schedule)
        model.compile(optimizer=optimizer, loss=CTCLoss, metrics=[CERMetric(
            greedy=args.greedy, beam_width=args.beam_width), WERMetric()])

    model.summary()
