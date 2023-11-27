# Imports

# > Standard library
import json
import logging
import os
import subprocess
import uuid

# > Local dependencies
from model_management import summarize_model


def get_git_hash():
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


def get_config(args, model):
    return {
        'git_hash': get_git_hash(),
        'args': vars(args),
        'model': summarize_model(model),
        'notes': ' ',
        'uuid': str(uuid.uuid4())
    }


def store_info(args, model):
    config = get_config(args, model)
    config_file_output = args.config_file_output if args.config_file_output \
        else os.path.join(args.output, 'config.json')

    try:
        with open(config_file_output, 'w') as configuration_file:
            json.dump(config, configuration_file)
    except IOError as e:
        logging.error(f"Error writing config file: {e}")
