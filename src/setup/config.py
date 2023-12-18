# Imports

# > Standard library
import argparse
import json
import logging
import os
import subprocess
import uuid


class Config:
    def __init__(self, args=None, default_args=None):
        self.default_args = default_args or argparse.Namespace()
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

    def __str__(self):
        return json.dumps(self.config, indent=4, sort_keys=True)

    def save(self, output_file=None):
        if not output_file:
            output_file = self.args.config_file_output or \
                f"{self.args.output}/config.json"
        try:
            with open(output_file, "w") as file:
                json.dump(self.config, file, indent=4, sort_keys=True)
        except IOError:
            logging.error(f"Could not write to {output_file}.")

    def organize_args(self, args):
        return {
            "general": {
                "gpu": args.gpu,
                "output": args.output,
                "output_charlist": args.output_charlist,
                "config_file": args.config_file,
                "config_file_output": args.config_file_output,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "charlist": args.charlist
            },
            "training": {
                "epochs": args.epochs,
                "width": args.width,
                "train_list": args.train_list,
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
                "augment": args.augment,
                "elastic_transform": args.elastic_transform,
                "random_crop": args.random_crop,
                "random_width": args.random_width,
                "distort_jpeg": args.distort_jpeg,
                "do_random_shear": args.do_random_shear,
                "do_blur": args.do_blur,
                "do_invert": args.do_invert,
                "do_binarize_otsu": args.do_binarize_otsu,
                "do_binarize_sauvola": args.do_binarize_sauvola
            },
            "decoding": {
                "greedy": args.greedy,
                "beam_width": args.beam_width,
                "num_oov_indices": args.num_oov_indices,
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
                "channels": args.channels
            }
        }

    def update_args_from_file(self, config_file):
        with open(config_file) as file:
            config = json.load(file)
            config_args = config.get("args", {})

            for key, value in config_args.items():
                for subkey, subvalue in value.items():
                    try:
                        # If the arg does not have the default value, it means
                        # it was set by the user. In this case, we don't want
                        # to override it.
                        if getattr(self.args, subkey) != \
                                self.default_args[subkey]:

                            # If it is also different from the value in the
                            # config file, we warn the user that we are
                            # overriding the value.
                            if getattr(self.args, subkey) != subvalue:
                                logging.info(
                                    f"Overriding {subkey} from config")
                        else:
                            setattr(self.args, subkey, subvalue)

                    except AttributeError:
                        logging.warning(f"Invalid argument: {subkey}. "
                                        f"Skipping...")
                        continue

    def change_arg(self, key, value):
        self.args.__setattr__(key, value)
        self.config["args"] = self.organize_args(self.args)

    def change_key(self, key, value):
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
