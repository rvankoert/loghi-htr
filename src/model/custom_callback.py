# Imports

# > Standard Library
import json
import logging
import os
import threading

# > Third party libraries
import tensorflow as tf

# > Local dependencies
from setup.config import Config


class LoghiCustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for TensorFlow Keras models to manage saving models and
    additional data during training.

    Parameters
    ----------
    save_best : bool
        If True, saves the model with the best validation metric.
    save_checkpoint : bool
        If True, saves the model at the end of each epoch.
    output : str
        Directory path to save the model and additional files.
    charlist : list of str, optional
        List of characters used in the model, saved alongside the model.
    config : object, optional
        Configuration object to be saved with the model.
    normalization_file : str, optional
        Path to the normalization file to be saved with the model.
    logging_level : str, default 'info'
        Logging level to be used in the callback.

    Attributes
    ----------
    best_val_metric : float
        The best validation metric observed during training. Initialized as
        infinity.
    logger : logging.Logger
        Logger object used for logging information.
    """

    def __init__(self, save_best: bool = True, save_checkpoint: bool = True,
                 output: str = "output", charlist: str = None,
                 config: Config = None, normalization_file: str = None,
                 logging_level: str = "info"):
        """
        Initialize the callback with provided configuration.
        """

        super().__init__()
        self.save_best = save_best
        self.save_checkpoint = save_checkpoint
        self.output = output
        self.charlist = charlist
        self.config = config
        self.normalization_file = normalization_file
        self.logging_level = logging_level
        self.best_val_metric = float("inf")
        self._setup_logging()

    def _setup_logging(self):
        """
        Set up the logging configuration for the callback.
        """
        logging.basicConfig(level=self.logging_level.upper())
        self.logger = logging.getLogger(__name__)

    def _async_save_model(self, subdir: str):
        """
        Save the model and additional data asynchronously.

        Parameters
        ----------
        subdir : str
            Subdirectory name under the main output directory to save the
            model.
        """
        threading.Thread(target=self._save_model, args=(subdir,)).start()

    def _save_model(self, subdir: str):
        """
        Save the model and additional data to the specified subdirectory.

        Parameters
        ----------
        subdir : str
            Subdirectory name under the main output directory to save the
            model.

        Raises
        ------
        Exception
            If there is an error during saving the model or additional data.
        """

        try:
            # Create output directory if necessary
            outputdir = os.path.join(self.output, subdir)
            os.makedirs(outputdir, exist_ok=True)
            model_path = os.path.join(outputdir, "model.keras")

            # Save model
            self.model.save(model_path)

            # Save additional files
            if self.charlist:
                with open(os.path.join(outputdir, "charlist.txt",
                                       encoding="utf-8"), "w") \
                        as chars_file:
                    chars_file.write("".join(self.charlist))
            if self.config:
                self.config.save(os.path.join(outputdir, "config.json"))
            if self.normalization_file:
                with open(self.normalization_file, "r", encoding="utf-8") \
                        as norm_file:
                    normalization = json.load(norm_file)
                with open(os.path.join(outputdir, "normalization.json"),
                          "w", encoding="utf-8") as norm_file:
                    json.dump(normalization, norm_file, indent=4,
                              ensure_ascii=False)

        except Exception as e:
            self.logger.error("Error saving model: %s", e)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Actions to perform at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The index of the epoch that just ended.
        logs : dict, optional
            A dictionary of logs from the training process.
        """

        logs = logs or {}
        current_val_metric = logs.get("val_CER_metric", None)

        # Log epoch metrics
        logging_text = f"Epoch {epoch} - Average CER: " \
                       f"{logs.get('CER_metric', 0):.4f}"
        if current_val_metric is not None:
            logging_text += f" - Validation CER: {current_val_metric:.4f}"
        print()
        self.logger.info(logging_text)

        # Save model if necessary
        if self.save_best and current_val_metric is not None:
            if current_val_metric < self.best_val_metric:
                self.logger.info("Validation CER improved from %.4f to %.4f",
                                 self.best_val_metric, current_val_metric)
                self.best_val_metric = current_val_metric
                self._async_save_model("best_val")

        # Save checkpoint
        if self.save_checkpoint:
            ckpt_name = f"epoch_{epoch}_CER_{logs.get('CER_metric', 0):.4f}"
            if current_val_metric is not None:
                ckpt_name += f"_val_{current_val_metric:.4f}"
            self.logger.info("Saving checkpoint...")
            self._async_save_model(ckpt_name)
            self.logger.info("Checkpoint saved.")
