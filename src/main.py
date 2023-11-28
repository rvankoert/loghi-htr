# Imports

# > Standard library
import os
import time
import logging

# > Local dependencies
from arg_parser import get_args
from config_metadata import store_info
from custom_layers import ResidualBlock
from data_handling import load_initial_charlist, initialize_data_loader, \
    save_charlist
from env_setup import setup_environment, setup_logging
from inference import perform_inference
from model import CERMetric, WERMetric, CTCLoss
from model_management import load_or_create_model, customize_model, \
    verify_charlist_length
from optimization import create_learning_rate_schedule, get_optimizer
from training import train_model, plot_training_history
from validation import perform_validation
from vgsl_model_generator import VGSLModelGenerator


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
    if args.existing_model:
        charlist = load_initial_charlist(
            args.charlist, args.existing_model,
            args.output, args.replace_final_layer)
    else:
        charlist = []

    # Set the custom objects
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(args, custom_objects)

        # Initialize the Dataloader
        loader = initialize_data_loader(args, charlist, model)
        training_dataset, validation_dataset, test_dataset, \
            inference_dataset, utilsObject, train_batches\
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
    timestamps = {'start_time': time.time()}

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
