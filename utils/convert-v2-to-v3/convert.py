import os

import logging
import shutil

import tensorflow as tf


from custom_layers import ResidualBlock
from losses import CTCLoss
from metrics import CERMetric, WERMetric
from optimization import LoghiLearningRateSchedule


def convert_savedmodel_to_keras(savedmodel_dir: str, output_directory: str, custom_objects: dict):
    """
    Converts a Keras SavedModel to the .keras format.

    Parameters
    ----------
    savedmodel_dir : str
        Path to the directory containing the SavedModel.
    output_directory : str
        Path to save the converted .keras file.
    custom_objects : dict

    Returns
    -------
    None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Validate input directory
    if not os.path.exists(savedmodel_dir):
        logging.error(f"SavedModel directory does not exist: {savedmodel_dir}")
        return

    if not os.path.exists(os.path.join(savedmodel_dir, "saved_model.pb")):
        logging.error(f"Invalid SavedModel directory: {savedmodel_dir}. Missing 'saved_model.pb'.")
        return

    try:
        # Load the SavedModel
        logging.info(f"Loading SavedModel from: {savedmodel_dir}")
        model = tf.keras.models.load_model(savedmodel_dir, custom_objects=custom_objects, compile=False)


        # Check if there is already a model.keras file in the output folder
        if os.path.exists(os.path.join(output_directory, 'model.keras')):
            logging.warning("A model.keras file already exists in the output "
                            "folder. Attempting to load the model from disk.")
            try:
                return tf.keras.models.load_model(
                    os.path.join(output_directory, 'model.keras'),
                    custom_objects=custom_objects)
            except Exception:
                raise ValueError(
                    "Failed to load the model from disk. Please clean the "
                    f"output folder '{output_directory}' and try again.")

        try:
            # Copy the model directory to the output folder
            shutil.copytree(savedmodel_dir, output_directory)
            shutil.rmtree(os.path.join(output_directory, 'assets'),
                          ignore_errors=True)
            shutil.rmtree(os.path.join(output_directory, 'variables'),
                          ignore_errors=True)

            for file in os.listdir(output_directory):
                if file.endswith('.pb'):
                    os.remove(os.path.join(output_directory, file))
        except Exception as e:
            logging.error(f"Error during file operations: {e}")
            raise

        model.save(os.path.join(output_directory, 'model.keras'))
        logging.info(
            f"Model converted and saved to {output_directory}/model.keras")

        # model.save(output_directory, save_format="keras")
        # logging.info("Model successfully converted to .keras format.")
    except Exception as e:
        logging.error(f"Failed to convert model: {e}")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert Keras SavedModel to .keras format.")
    parser.add_argument("--savedmodel_dir", type=str, required=True, help="Path to the SavedModel directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the .keras file.")

    args = parser.parse_args()

    custom_objects = {'CERMetric': CERMetric, 'WERMetric': WERMetric,
                      'CTCLoss': CTCLoss, 'ResidualBlock': ResidualBlock,
                      'LoghiLearningRateSchedule': LoghiLearningRateSchedule}

    # Convert the model
    convert_savedmodel_to_keras(args.savedmodel_dir, args.output_file, custom_objects)