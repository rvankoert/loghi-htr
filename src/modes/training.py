# Imports

# > Standard library
import os
from typing import Any

# > Third-party dependencies
import matplotlib.pyplot as plt
import tensorflow as tf

# > Local dependencies
from data.loader import DataLoader
from setup.config_metadata import get_config
from model.model import train_batch


def train_model(model: tf.keras.Model,
                args: Any,
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
