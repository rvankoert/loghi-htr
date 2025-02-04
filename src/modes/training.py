# Imports

# > Standard library
import os

# > Third-party dependencies
import matplotlib.pyplot as plt
import tensorflow as tf

# > Local dependencies
from data.manager import DataManager
from setup.config import Config
from model.custom_callback import LoghiCustomCallback


def train_model(model: tf.keras.Model,
                config: Config,
                training_dataset: tf.data.Dataset,
                validation_dataset: tf.data.Dataset,
                data_manager: DataManager) -> tf.keras.callbacks.History:
    """
    Trains a Keras model using the provided training and validation datasets,
    along with additional arguments.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    config : Config
        A Config object containing model and training configurations.
    training_dataset : tf.data.Dataset
        The dataset to be used for training.
    validation_dataset : tf.data.Dataset
        The dataset to be used for validation.
    data_manager : DataManager
        A DataManager containing additional information like character list.

    Returns
    -------
    tf.keras.callbacks.History
        The training history object containing information about the training
        process (e.g., loss values, metrics).

    Notes
    -----
    This function sets up a custom training routine for a Keras model, with
    logging and early stopping functionalities. The actual training process
    depends on the specific model and data provided.
    """

    # CSV logger
    log_filename = os.path.join(config["output"], 'log.csv')
    logging_callback = tf.keras.callbacks.CSVLogger(
        log_filename, separator=",", append=True)

    # Loghi custom callback
    loghi_custom_callback = \
        LoghiCustomCallback(save_best=True,
                            save_checkpoint=config["output_checkpoints"],
                            output=config["output"],
                            tokenizer=data_manager.tokenizer,
                            config=config,
                            normalization_file=config["normalization_file"])

    # Add all default callbacks
    callbacks = [logging_callback, loghi_custom_callback]

    # If we defined an early stopping patience, add it to the callbacks
    if config["early_stopping_patience"] > 0 and validation_dataset:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_CER_metric',
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)

    # Determine the number of steps per epoch
    # NOTE: None means that the number of steps is equal to the number of
    # batches in the dataset (default behavior)
    # FIXME: steps_per_epoch is not working properly
    steps_per_epoch = config["steps_per_epoch"] \
        if config["steps_per_epoch"] else None

    # Train the model
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=config["epochs"],
        callbacks=callbacks,
        shuffle=True,
        steps_per_epoch=steps_per_epoch,
        verbose=config["training_verbosity_mode"]
    )

    return history


def plot_metric(metric, history, title, output_path, plot_validation_metric):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history[metric], label='Training ' + metric)
    if plot_validation_metric:
        plt.plot(history.history[f"val_{metric}"],
                 label=f"Validation {metric}")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel(metric)
    plt.legend(loc="upper right")
    plt.savefig(output_path)
    plt.close()


def plot_training_history(history: tf.keras.callbacks.History,
                          output_path: str,
                          plot_validation: bool = False) -> None:
    """
    Plots the training history of a Keras model, including loss and Character
    Error Rate (CER).

    Parameters
    ----------
    history : tf.keras.callbacks.History
        The training history object returned by a model training process,
        containing metrics like loss and CER over epochs.
    output_path : str
        Path to save the plots.
    plot_validation : bool, default False
        Whether to plot validation metrics.

    Notes
    -----
    This function generates and saves two plots: one for training loss and the
    other for Character Error Rate (CER).
    """

    plot_metric(metric="loss",
                history=history,
                title="Training Loss",
                output_path=os.path.join(output_path, 'loss_plot.png'),
                plot_validation_metric=plot_validation)
    plot_metric(metric="CER_metric",
                history=history,
                title="Character Error Rate (CER)",
                output_path=os.path.join(output_path, 'cer_plot.png'),
                plot_validation_metric=plot_validation)
