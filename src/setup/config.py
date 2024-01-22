# Imports

# > Standard library
import argparse
import json
import logging
import os
import subprocess
import uuid


class Config:
    """
    A class for managing configuration settings for an application.

    This class handles the loading, organizing, and saving of configuration
    parameters. It supports reading from a file and writing to a file, as well
    as updating parameters at runtime.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command line arguments provided by the user.
    default_args : argparse.Namespace, optional
        Default command line arguments.

    Attributes
    ----------
    default_args : argparse.Namespace
        Default command line arguments.
    args : argparse.Namespace
        Command line arguments provided by the user.
    git_hash : str
        Git hash of the current codebase.
    notes : str
        Custom notes or comments.
    uuid : str
        Unique identifier for the configuration.
    url_code : str
        URL of the code repository.
    config : dict
        Dictionary containing organized configuration settings.
    """

    def __init__(self, args: argparse.Namespace = None,
                 explicit_args: argparse.Namespace = None):
        """
        Initialize the Config object with provided arguments and default
        arguments.

        Parameters
        ----------
        args : argparse.Namespace, optional
            Command line arguments provided by the user.
        default_args : argparse.Namespace, optional
            Default command line arguments.
        """

        self.explicit_args = explicit_args or argparse.Namespace()
        self.args = args or argparse.Namespace()
        if self.args.config_file:
            self.update_args_from_file(self.args.config_file)

        self.git_hash = get_git_hash()
        self.notes = ""
        self.uuid = str(uuid.uuid4())
        self.url_code = "https://github.com/knaw-huc/loghi"
        self.config = {"args": self.organize_args(self.args),
                       "git_hash": self.git_hash,
                       "notes": self.notes,
                       "uuid": self.uuid,
                       "url_code": self.url_code}

    def __str__(self) -> str:
        """
        String representation of the Config object.

        Returns
        -------
        str
            JSON string representation of the config dictionary.
        """

        return json.dumps(self.config, indent=4, sort_keys=True)

    def __getitem__(self, key: str) -> any:
        """
        Get a value from the configuration dictionary.

        Parameters
        ----------
        key : str
            The key of the value to retrieve.

        Returns
        -------
        any
            The value corresponding to the key.
        """

        try:
            return self.config[key]
        except KeyError:
            return getattr(self.args, key, None)

    def save(self, output_file: str = None) -> None:
        """
        Save the configuration settings to a file.

        Parameters
        ----------
        output_file : str, optional
            The path of the file where the configuration will be saved.
        """

        if not output_file:
            output_file = self.args.config_file_output or \
                f"{self.args.output}/config.json"
        try:
            with open(output_file, "w") as file:
                json.dump(self.config, file, indent=4, sort_keys=True)
        except IOError:
            logging.error(f"Could not write to {output_file}.")

    def organize_args(self, args: argparse.Namespace) -> dict:
        """
        Organize arguments into a structured dictionary.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to organize.

        Returns
        -------
        dict
            A dictionary with organized arguments categorized into
            sub-dictionaries.
        """

        return {
            "general": {
                "gpu": args.gpu,
                "output": args.output,
                "config_file": args.config_file,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "charlist": args.charlist
            },
            "training": {
                "epochs": args.epochs,
                "width": args.width,
                "train_list": args.train_list,
                "test_list": args.test_list,
                "steps_per_epoch": args.steps_per_epoch,
                "output_checkpoints": args.output_checkpoints,
                "early_stopping_patience": args.early_stopping_patience,
                "do_validate": args.do_validate,
                "validation_list": args.validation_list,
                "training_verbosity_mode": args.training_verbosity_mode,
                "max_queue_size": args.max_queue_size
            },
            "inference": {
                "inference_list": args.inference_list,
                "results_file": args.results_file
            },
            "learning_rate": {
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "decay_rate": args.decay_rate,
                "decay_steps": args.decay_steps,
                "warmup_ratio": args.warmup_ratio,
                "decay_per_epoch": args.decay_per_epoch,
                "linear_decay": args.linear_decay
            },
            "model": {
                "model": args.model,
                "use_float32": args.use_float32,
                "existing_model": args.existing_model,
                "model_name": args.model_name,
                "replace_final_layer": args.replace_final_layer,
                "replace_recurrent_layer": args.replace_recurrent_layer,
                "thaw": args.thaw,
                "freeze_conv_layers": args.freeze_conv_layers,
                "freeze_recurrent_layers": args.freeze_recurrent_layers,
                "freeze_dense_layers": args.freeze_dense_layers
            },
            "augmentation": {
                "multiply": args.multiply,
                "elastic_transform": args.elastic_transform,
                "random_crop": args.random_crop,
                "random_width": args.random_width,
                "distort_jpeg": args.distort_jpeg,
                "do_random_shear": args.do_random_shear,
                "do_blur": args.do_blur,
                "do_invert": args.do_invert,
                "do_binarize_otsu": args.do_binarize_otsu,
                "do_binarize_sauvola": args.do_binarize_sauvola,
                "visualize_augments": args.visualize_augments
            },
            "decoding": {
                "greedy": args.greedy,
                "beam_width": args.beam_width,
                "corpus_file": args.corpus_file,
                "wbs_smoothing": args.wbs_smoothing
            },
            "misc": {
                "ignore_lines_unknown_character":
                    args.ignore_lines_unknown_character,
                "check_missing_files": args.check_missing_files,
                "normalization_file": args.normalization_file,
                "deterministic": args.deterministic
            },
            "depr": {
                "do_train": args.do_train,
                "do_inference": args.do_inference,
                "use_mask": args.use_mask,
                "no_auto": args.no_auto,
                "height": args.height,
                "channels": args.channels,
                "output_charlist": args.output_charlist,
                "config_file_output": args.config_file_output,
            }
        }

    def update_args_from_file(self, config_file: str) -> None:
        """
        Update arguments from a configuration file.

        Parameters
        ----------
        config_file : str
            Path to the configuration file to load arguments from.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        """

        with open(config_file) as file:
            config = json.load(file)
            config_args = config.get("args", {})

            if not config_args:
                logging.warning("No arguments found in config file.")
                return

            # For backwards compatibility, we check if the config file has
            # a "general" key. If not, we assume that all arguments are
            # general arguments.
            if not isinstance(config_args.get("general", None), dict):
                logging.warning("No general arguments found in config file. "
                                "Assuming v1 config file.")
                config_args = {"general": config_args}

            for key, value in config_args.items():
                for subkey, subvalue in value.items():
                    try:
                        # If the argument was explicitly provided by the user,
                        # we do not override it. Otherwise, we update it.
                        if not getattr(self.explicit_args, subkey):
                            setattr(self.args, subkey, subvalue)
                        else:
                            # If the argument is the same as the config file,
                            # There is no need to log it.
                            if getattr(self.args, subkey) == subvalue:
                                continue
                            logging.info("Overriding config file argument "
                                         f"'{subkey}' with command line "
                                         "argument.")

                    except AttributeError:
                        logging.warning(f"Invalid argument: {subkey}. "
                                        f"Skipping...")
                        continue

    def change_arg(self, key: str, value: any) -> None:
        """
        Change a specific argument's value.

        Parameters
        ----------
        key : str
            The key of the argument to change.
        value : any
            The new value to assign to the argument.
        """

        setattr(self.args, key, value)
        self.config["args"] = self.organize_args(self.args)

    def update_config_key(self, key: str, value: any) -> None:
        """
        Change a specific key in the configuration dictionary.

        Parameters
        ----------
        key : str
            The key of the configuration to change.
        value : any
            The new value to assign to the key.
        """

        self.config[key] = value


def get_git_hash() -> str:
    """
    Retrieves the current Git commit hash of the codebase.

    Returns
    -------
    str
        The Git commit hash if available; otherwise, returns 'Unavailable'.

    Notes
    -----
    The function first checks for a 'version_info' file and reads the hash from
    there. If not found, it tries to retrieve the hash using the 'git' command.
    It handles subprocess and OS errors, logging them if they occur.
    """

    if os.path.exists("version_info"):
        with open("version_info") as file:
            return file.read().strip()
    else:
        try:
            result = subprocess.run(['git', 'log', '--format=%H', '-n', '1'],
                                    stdout=subprocess.PIPE,
                                    check=True)
            return result.stdout.decode('utf-8').strip().replace('"', '')
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess failed: {e}")
        except OSError as e:
            logging.error(f"OS error occurred: {e}")
        return "Unavailable"
