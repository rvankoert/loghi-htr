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
from modes.inference import perform_inference
from model.model import CERMetric, WERMetric, CTCLoss
from model.management import load_or_create_model, customize_model, \
    verify_charlist_length
from model.optimization import create_learning_rate_schedule, get_optimizer
from modes.training import train_model, plot_training_history
from modes.validation import perform_validation

# Setup and configuration
from setup.arg_parser import get_args
from setup.config import Config
from setup.environment import setup_environment, setup_logging

from utils.print import summarize_model


def main():
    setup_logging()

    # Get the arguments
    args, default_args = get_args()
    config = Config(args, default_args)
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
        training_dataset, validation_dataset, test_dataset, \
            inference_dataset, utilsObject, train_batches \
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
    config.change_key("model", summarize_model(model))
    config.change_arg("channels", model.input_shape[-1])
    config.save()

    # Store timestamps
    timestamps = {'start_time': time.time()}

    # Train the model
    if args.do_train:
        tick = time.time()

        history = train_model(model, config, training_dataset,
                              validation_dataset, loader,
                              lr_schedule)

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
        tick = time.time()
        perform_inference(args, model, inference_dataset, charlist, loader)

        timestamps['inference_time'] = time.time() - tick

    # Log the timestamps
    logging.info("--------------------------------------------------------")
    if args.do_train:
        logging.info(f"Training completed in {timestamps['train_time']:.2f} "
                     "seconds")
    if args.do_validate:
        logging.info("Validation completed in "
                     f"{timestamps['validate_time']:.2f} seconds")
    if args.do_inference:
        logging.info("Inference completed in "
                     f"{timestamps['inference_time']:.2f} seconds")
    logging.info(f"Total time: {time.time() - timestamps['start_time']:.2f} "
                 "seconds")


if __name__ == "__main__":
    main()
