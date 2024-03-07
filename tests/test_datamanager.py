# Imports

# > Standard library
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

# > Third party dependencies


class DataManagerTest(unittest.TestCase):
    """
    Tests for the DataManager class.

    Test Coverage:
        1. `test_initialization`: Checks the correct instantiation of the
        DataManager class and its default values.
        2. `test_create_data_simple`: Checks if the create_data function
        works as expected.
        3. `test_missing_files`: Tests the behavior when data lists contain
        references to non-existent files.
        4. `test_unsupported_chars`: Validates the handling of labels with
        unsupported characters.
        5. `test_inference_mode`: Checks the DataManager"s behavior in
        inference mode.
        6. `test_text_normalization`: Verifies the correct normalization of
        text labels.
        7. `test_multiplication`: Tests the multiplication functionality for
        increasing dataset size.
        8. `test_generators`: Validates the creation and behavior of data
        generators for training, validation, test, and inference.
    """

    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
            level=logging.ERROR,
        )

        cls.dummy_config = {
            "do_validate": False,
            "train_list": "",
            "validation_list": "",
            "test_list": "",
            "inference_list": "",
            "replace_final_layer": False,
            "use_mask": False,
        }

        # Determine the directory of this file
        current_file_dir = Path(__file__).resolve().parent

        # Get the project root
        if current_file_dir.name == "tests":
            project_root = current_file_dir.parent
        else:
            project_root = current_file_dir

        # Add 'src' directory to sys.path
        sys.path.append(str(project_root / 'src'))

        # Set paths for data and model directories
        cls.data_dir = project_root / "tests" / "data"

        cls.sample_image_paths = [os.path.join(
            cls.data_dir, f"test-image{i+1}") for i in range(3)]

        # Extract labels from .txt files
        cls.sample_labels = []
        for i in range(3):
            label_path = os.path.join(cls.data_dir, f"test-image{i+1}.txt")
            with open(label_path, "r") as file:
                cls.sample_labels.append(file.readline().strip())

        # Create sample list file
        cls.sample_list_file = os.path.join(cls.data_dir, "sample_list.txt")
        with open(cls.sample_list_file, "w") as f:
            for img_path, label in zip(cls.sample_image_paths,
                                       cls.sample_labels):
                f.write(f"{img_path}.png\t{label}\n")

        from data.manager import DataManager
        cls.DataManager = DataManager

        from utils.text import Tokenizer
        cls.Tokenizer = Tokenizer

    def _create_temp_file(self, additional_lines=None):
        temp_sample_list_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+")

        for img_path, label in zip(self.sample_image_paths,
                                   self.sample_labels):
            temp_sample_list_file.write(f"{img_path}.png\t{label}\n")
        if additional_lines:
            for line in additional_lines:
                temp_sample_list_file.write(line + "\n")

        temp_sample_list_file.close()
        return temp_sample_list_file.name

    def _remove_temp_file(self, filename):
        os.remove(filename)

    def test_initialization(self):
        # Only provide the required arguments for initialization and check them
        test_config = self.dummy_config.copy()
        test_config.update({
            "batch_size": 32,
            "img_size": (256, 256, 3),
        })

        data_manager = self.DataManager(img_size=test_config["img_size"],
                                        config=test_config,
                                        augment_model=None,
                                        charlist=["a", "b", "c"])
        self.assertIsInstance(data_manager, self.DataManager,
                              "DataManager not instantiated correctly")


if __name__ == "__main__":
    unittest.main()
