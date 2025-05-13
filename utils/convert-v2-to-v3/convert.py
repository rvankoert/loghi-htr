import os

import logging
import tensorflow as tf


from custom_layers import ResidualBlock
from losses import CTCLoss
from metrics import CERMetric, WERMetric
from optimization import LoghiLearningRateSchedule


def convert_savedmodel_to_keras(savedmodel_dir: str, output_file: str, custom_objects: dict):
    """
    Converts a Keras SavedModel to the .keras format.

    Parameters
    ----------
    savedmodel_dir : str
        Path to the directory containing the SavedModel.
    output_file : str
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

        # Save the model in .keras format
        logging.info(f"Saving model to .keras format at: {output_file}")
        model.save(output_file, save_format="keras")
        logging.info("Model successfully converted to .keras format.")
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