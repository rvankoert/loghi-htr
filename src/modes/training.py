# Imports

# > Standard library
import os

# > Third-party dependencies
import matplotlib.pyplot as plt

# > Local dependencies
from setup.config_metadata import get_config
from model.model import train_batch


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
