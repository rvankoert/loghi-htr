# Imports

# > Standard library
import os
from typing import Any, List, Optional

# > Third-party dependencies
import matplotlib.pyplot as plt
import tensorflow as tf

# > Local dependencies
from data.loader import DataLoader
from setup.config import Config
from model.custom_callback import LoghiCustomCallback


def train_model(model: tf.keras.Model,
                config: Config,
                training_dataset: tf.data.Dataset,
                validation_dataset: tf.data.Dataset,
                loader: DataLoader) -> Any:
    """
    Trains a Keras model using the provided training and validation datasets,
    along with additional arguments.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    args : Any
        A set of arguments containing training parameters such as epochs,
        output paths, steps per epoch, etc.
    training_dataset : tf.data.Dataset
        The dataset to be used for training.
    validation_dataset : tf.data.Dataset
        The dataset to be used for validation.
    loader : DataLoader
        A DataLoader containing additional information like character list.

    Returns
    -------
    Any
        The training history object containing information about the training
        process (e.g., loss values, metrics).

    Notes
    -----
    This function uses the `train_batch` function internally to train the
    model, and handles various aspects of training like early stopping,
    checkpointing, and setting up metadata.
    """
    args = config.args

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
        config=config,
        verbosity_mode=args.training_verbosity_mode,
        normalization_file=args.normalization_file
    )

    return history


def train_batch(model: tf.keras.Model,
                train_dataset: tf.data.Dataset,
                validation_dataset: Optional[tf.data.Dataset],
                epochs: int,
                output: str,
                model_name: str,
                steps_per_epoch: Optional[int] = None,
                early_stopping_patience: int = 20,
                num_workers: int = 20,
                max_queue_size: int = 256,
                output_checkpoints: bool = False,
                config: Config = None,
                charlist: Optional[List[str]] = None,
                verbosity_mode: str = 'auto',
                normalization_file: str = None) -> tf.keras.callbacks.History:
    """
    Train a given Keras model using specified datasets and training
    configurations.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    train_dataset : tf.data.Dataset
        The training dataset.
    validation_dataset : tf.data.Dataset, optional
        The validation dataset. If not provided, validation is skipped.
    epochs : int
        Number of epochs to train the model.
    output : str
        Directory path to save training outputs.
    model_name : str
        Name of the model.
    steps_per_epoch : int, optional
        Number of steps per epoch. If not provided, it's inferred from the
        dataset.
    early_stopping_patience : int, default 20
        Number of epochs with no improvement after which training will be
        stopped.
    num_workers : int, default 20
        Number of workers for data loading.
    max_queue_size : int, default 256
        Maximum size for the generator queue.
    output_checkpoints : bool, default False
        Whether to output model checkpoints.
    config : Config, optional
        A Config object containing additional metadata.
    charlist : list of str, optional
        List of characters involved in the training process.
    verbosity_mode : str, default 'auto'
        Verbosity mode, 'auto', 'silent', or 'verbose'.
    normalization_file : str, optional
        Path to the normalization file.

    Returns
    -------
    tf.keras.callbacks.History
        Training history object.

    Notes
    -----
    This function sets up a custom training routine for a Keras model, with
    logging and early stopping functionalities. The actual training process
    depends on the specific model and data provided.
    """

    # CSV logger
    log_filename = os.path.join(output, 'log.csv')
    logging_callback = tf.keras.callbacks.CSVLogger(
        log_filename, separator=",", append=True)

    # Loghi custom callback
    loghi_custom_callback = \
        LoghiCustomCallback(save_best=True,
                            save_checkpoint=output_checkpoints,
                            output=output,
                            charlist=charlist,
                            config=config,
                            normalization_file=normalization_file)

    # Add all default callbacks
    callbacks = [logging_callback, loghi_custom_callback]

    # If we defined an early stopping patience, add it to the callbacks
    if early_stopping_patience > 0 and validation_dataset:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_CER_metric',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='min'
        )
        callbacks.append(early_stopping)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        workers=num_workers,
        max_queue_size=max_queue_size,
        steps_per_epoch=steps_per_epoch,
        verbose=verbosity_mode
    )
    return history


def plot_training_history(history: Any, args: Any) -> None:
    """
    Plots the training history of a Keras model, including loss and Character
    Error Rate (CER).

    Parameters
    ----------
    history : Any
        The training history object returned by a model training process,
        containing metrics like loss and CER over epochs.
    args : Any
        A set of arguments that includes validation information and output
        paths for saving plots.

    Notes
    -----
    This function generates and saves two plots: one for training loss and the
    other for Character Error Rate (CER). It checks for validation data
    availability in `args` to include validation metrics in the plots.
    """

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
