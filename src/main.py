# Imports

# > Standard library
import os
import time
import logging

# > Local dependencies
# Data handling
from data.data_handling import load_initial_charlist, initialize_data_loader, \
    save_charlist

# Model-specific
from model.custom_layers import ResidualBlock
from model.losses import CTCLoss
from model.metrics import CERMetric, WERMetric
from model.management import load_or_create_model, customize_model, \
    verify_charlist_length
from model.optimization import create_learning_rate_schedule, get_optimizer
from modes.training import train_model, plot_training_history
from modes.validation import perform_validation
from modes.test import perform_test
from modes.inference import perform_inference

# Setup and configuration
from setup.arg_parser import get_args
from setup.config import Config
from setup.environment import setup_environment, setup_logging

# Utilities
from utils.print import summarize_model


def main():
    setup_logging()

    # Get the arguments
    parsed_args = get_args()
    config = Config(*parsed_args)
    args = config.args

    # Set up the environment
    strategy = setup_environment(config)

    # Create the output directory if it doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Get the initial character list
    if args.existing_model:
        charlist = load_initial_charlist(
            args.charlist, args.existing_model,
            args.output, args.replace_final_layer)
    else:
        charlist = []

    # Set the custom objects
    from model.optimization import LoghiLearningRateSchedule
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(args, custom_objects)

        # Initialize the Dataloader
        loader = initialize_data_loader(args, charlist, model)
        training_dataset, evaluation_dataset, validation_dataset, \
            test_dataset, inference_dataset, tokenizer, train_batches, \
            validation_labels = loader.generators()

        # Replace the charlist with the one from the data loader
        charlist = loader.charList

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, args, charlist)

        # Save the charlist
        verify_charlist_length(charlist, model, args.use_mask)
        save_charlist(charlist, args.output)

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            learning_rate=args.learning_rate,
            decay_rate=args.decay_rate,
            decay_steps=args.decay_steps,
            train_batches=train_batches,
            do_train=args.do_train,
            warmup_ratio=args.warmup_ratio,
            epochs=args.epochs,
            decay_per_epoch=args.decay_per_epoch,
            linear_decay=args.linear_decay)

        # Create the optimizer
        optimizer = get_optimizer(args.optimizer, lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer, loss=CTCLoss, metrics=[CERMetric(
            greedy=args.greedy, beam_width=args.beam_width), WERMetric()])

    # Print the model summary
    model.summary()

    # Store the model info (i.e., git hash, args, model summary, etc.)
    config.update_config_key("model", summarize_model(model))
    config.update_config_key("model_name", model.name)
    config.update_config_key("model_channels", model.input_shape[-1])
    config.save()

    # Store timestamps
    timestamps = {'start_time': time.time()}

    # Train the model
    if args.do_train:
        tick = time.time()

        history = train_model(model, config, training_dataset,
                              evaluation_dataset, loader)

        # Plot the training history
        plot_training_history(history, args.output,
                              True if args.validation_list else False)

        timestamps['Training'] = time.time() - tick

    # Evaluate the model
    if args.do_validate:
        logging.warning("Validation results are without special markdown tags")

        tick = time.time()
        perform_validation(config, model, validation_dataset,
                           validation_labels, charlist, loader)
        timestamps['Validation'] = time.time() - tick

    # Test the model
    if args.test_list:
        logging.warning("Test results are without special markdown tags")

        tick = time.time()
        perform_test(config, model, test_dataset, charlist, loader)
        timestamps['Test'] = time.time() - tick

    # Infer with the model
    if args.do_inference:
        tick = time.time()
        perform_inference(config, model, inference_dataset, charlist, loader)
        timestamps['Inference'] = time.time() - tick

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    for key, value in list(timestamps.items())[1:]:
        logging.info(f"{key} completed in {value:.2f} seconds")
    logging.info(f"Total time: {time.time() - timestamps['start_time']:.2f} "
                 "seconds")


if __name__ == "__main__":
    main()
