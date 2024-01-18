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

    # Set up the environment
    strategy = setup_environment(config)

    # Create the output directory if it doesn't exist
    if config["output"]:
        os.makedirs(config["output"], exist_ok=True)

    # Get the initial character list
    if config["existing_model"] or config["charlist"]:
        charlist, removed_padding = load_initial_charlist(
            config["charlist"], config["existing_model"],
            config["output"], config["replace_final_layer"])
    else:
        charlist = []
        removed_padding = False

    # Set the custom objects
    from model.optimization import LoghiLearningRateSchedule
    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Create the model
    with strategy.scope():
        model = load_or_create_model(config, custom_objects)

        # Initialize the Dataloader
        loader = initialize_data_loader(config, charlist, model)
        training_dataset, evaluation_dataset, validation_dataset, \
            test_dataset, inference_dataset, tokenizer, train_batches, \
            validation_labels = loader.generators()

        # Replace the charlist with the one from the data loader
        charlist = loader.charList

        # Additional model customization such as freezing layers, replacing
        # layers, or adjusting for float32
        model = customize_model(model, config, charlist)

        # Save the charlist
        verify_charlist_length(charlist, model, config["use_mask"],
                               removed_padding)
        save_charlist(charlist, config["output"])

        # Create the learning rate schedule
        lr_schedule = create_learning_rate_schedule(
            learning_rate=config["learning_rate"],
            decay_rate=config["decay_rate"],
            decay_steps=config["decay_steps"],
            train_batches=train_batches,
            do_train=config["do_train"],
            warmup_ratio=config["warmup_ratio"],
            epochs=config["epochs"],
            decay_per_epoch=config["decay_per_epoch"],
            linear_decay=config["linear_decay"])

        # Create the optimizer
        optimizer = get_optimizer(config["optimizer"], lr_schedule)

        # Compile the model
        model.compile(optimizer=optimizer, loss=CTCLoss,
                      metrics=[CERMetric(greedy=config["greedy"],
                                         beam_width=config["beam_width"]),
                               WERMetric()])

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
    if config["training_list"]:
        tick = time.time()

        history = train_model(model, config, training_dataset,
                              evaluation_dataset, loader)

        # Plot the training history
        plot_training_history(history, config["output"],
                              True if config["validation_list"] else False)

        timestamps['Training'] = time.time() - tick

    # Evaluate the model
    if config["do_validate"]:
        logging.warning("Validation results are without special markdown tags")

        tick = time.time()
        perform_validation(config, model, validation_dataset,
                           validation_labels, charlist, loader)
        timestamps['Validation'] = time.time() - tick

    # Test the model
    if config["test_list"]:
        logging.warning("Test results are without special markdown tags")

        tick = time.time()
        perform_test(config, model, test_dataset, charlist, loader)
        timestamps['Test'] = time.time() - tick

    # Infer with the model
    if config["inference_list"]:
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
