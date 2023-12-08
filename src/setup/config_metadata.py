# Imports

# > Standard library
import argparse
import json
import logging
import os
import subprocess
from typing import Any, Dict
import uuid

# > Third-party dependencies
import tensorflow as tf

# > Local dependencies
from utils.print import summarize_model


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


def get_config(args: argparse.Namespace, model: tf.keras.Model) \
        -> Dict[str, Any]:
    """
    Generates a configuration dictionary containing various details about the
    model and runtime arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace containing runtime arguments.
    model : tf.keras.Model
        The Keras model from which configuration details are derived.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the Git hash, runtime arguments, model summary,
        a generated UUID, and other details.

    Notes
    -----
    This function is used to collate various pieces of information about the
    model and the environment, such as the Git commit hash, the arguments
    passed to the script, and a summarized model description. It also generates
    a unique identifier (UUID) for the configuration.
    """

    args.channels = model.layers[0].input_shape[0][2]
    return {
        'git_hash': get_git_hash(),
        'args': vars(args),
        'model': summarize_model(model),
        'notes': ' ',
        'uuid': str(uuid.uuid4()),
        'url-code': 'https://github.com/knaw-huc/loghi'
    }


def store_info(args: argparse.Namespace, model: tf.keras.Model) -> None:
    """
    Stores the configuration information as a JSON file.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace containing runtime arguments including the output
        directory and optional specific output location for the config file.
    model : tf.keras.Model
        The Keras model from which configuration details are derived.

    Notes
    -----
    This function generates a configuration dictionary using `get_config` and
    writes it to a JSON file. The location of the file is determined by
    `args.config_file_output` or defaults to a 'config.json' file in the output
    directory. It handles and logs file IO errors.
    """

    config = get_config(args, model)
    config_file_output = args.config_file_output if args.config_file_output \
        else os.path.join(args.output, 'config.json')

    try:
        with open(config_file_output, 'w') as configuration_file:
            json.dump(config, configuration_file)
    except IOError as e:
        logging.error(f"Error writing config file: {e}")
