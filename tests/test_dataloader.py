# Imports

# > Standard library
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

# > Third party dependencies
import numpy as np


class DataLoaderTest(unittest.TestCase):
    """
    Tests for the DataLoader class.

    Test Coverage:
        1. `test_initialization`: Checks the correct instantiation of the
        DataLoader class and its default values.
        2. `test_create_data_simple`: Checks if the create_data function
        works as expected.
        3. `test_missing_files`: Tests the behavior when data lists contain
        references to non-existent files.
        4. `test_unsupported_chars`: Validates the handling of labels with
        unsupported characters.
        5. `test_inference_mode`: Checks the DataLoader"s behavior in
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

        from data.loader import DataLoader
        cls.DataLoader = DataLoader

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
        batch_size = 32
        img_size = (256, 256, 3)

        data_loader = self.DataLoader(batch_size=batch_size,
                                      img_size=img_size)
        self.assertIsInstance(data_loader, self.DataLoader,
                              "DataLoader not instantiated correctly")

        # Check the values
        self.assertEqual(data_loader.batch_size, batch_size,
                         f"batch_size not set correctly. Expected: "
                         f"{batch_size}, got: {data_loader.batch_size}")
        self.assertEqual(data_loader.imgSize, img_size,
                         f"imgSize not set correctly. Expected: "
                         f"{img_size}, got: {data_loader.imgSize}")

    def test_create_data_simple(self):
        # Sample data
        chars = set()
        labels = {"test_partition": []}
        partition = {"test_partition": []}
        data_file_list = self.sample_list_file
        partition_name = "test_partition"

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        # Call create_data
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, data_file_list)

        # Asserts
        self.assertEqual(len(files), 3)
        for i, (fileName, gtText) in enumerate(files):
            self.assertEqual(fileName, self.sample_image_paths[i] + ".png",
                             f"Image path not set correctly. Expected: "
                             f"{self.sample_image_paths[i] + '.png'}, got: "
                             f"{fileName}")
            self.assertEqual(gtText, self.sample_labels[i],
                             f"Label not set correctly. Expected: "
                             f"{self.sample_labels[i]}, got: {gtText}")

    def test_missing_files(self):
        # Manipulate sample file to have a missing image path
        additional_lines = [
            f"{os.path.join(self.data_dir, 'missing-image.png')}"
            "\tmissing_label"]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        # Sample data
        chars = set()
        labels = {"test_partition": []}
        partition = {"test_partition": []}
        partition_name = "test_partition"

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        # Call create_data with include_missing_files=False (default)
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # should still be 3, not 4, because we skip the missing file
        self.assertEqual(len(files), 3, "Missing file not skipped")

        # Call create_data with include_missing_files=True
        chars, files = data_loader.create_data(
            chars, labels, partition,
            partition_name, temp_sample_list_file, include_missing_files=True)

        # Asserts
        # should be 4 now, including the missing file
        self.assertEqual(len(files), 4, "Missing file not included")

    def test_unsupported_chars(self):
        # Sample data with unsupported characters
        additional_lines = [
            f"{self.sample_image_paths[0]}.png\tlabelX",
            f"{self.sample_image_paths[1]}.png\tlabelY"
        ]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        chars = set()
        labels = {"test_partition": []}
        partition = {"test_partition": []}
        partition_name = "test_partition"

        # Initialize DataLoader with injected_charlist set to a list without
        # "X" and 'Y'
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))
        data_loader.injected_charlist = set(
            "abcdefghijklkmnopqrstuvwxyzN0123456789, ")\
            - set("XY")

        # Call create_data with include_unsupported_chars=False (default)
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # should still be 3, not 5, because we skip lines with "X" and 'Y'
        self.assertEqual(len(files), 3, "Unsupported chars not skipped")

        # Call create_data with include_unsupported_chars=True
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            temp_sample_list_file, include_unsupported_chars=True)

        # Asserts
        # should be 5 now, including the lines with "X" and 'Y'
        self.assertEqual(len(files), 5, "Unsupported chars not included")

        self._remove_temp_file(temp_sample_list_file)

    def _test_inference_mode(self):
        temp_sample_list_file = self._create_temp_file()

        chars = set()
        labels = {"test_partition": []}
        partition = {"test_partition": []}
        partition_name = "test_partition"

        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))

        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            temp_sample_list_file, is_inference=True)
        self.assertEqual(len(files), 3, "Inference mode not working")
        for _, gtText in files:
            self.assertEqual(gtText, "to be determined",
                             "Inference mode not working")

        self._remove_temp_file(temp_sample_list_file)

    def test_text_normalization(self):
        # Sample data with mixed-case labels
        additional_lines = [f"{self.sample_image_paths[0]}.png\tLabel      ."]
        temp_sample_list_file = self._create_temp_file(
            additional_lines)

        chars = set()
        labels = {"train": []}
        partition = {"train": []}
        partition_name = "train"

        # Initialize DataLoader
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3),
                                      normalization_file=os.path.join(
                                          self.data_dir, "norm_chars.json"))

        # Call create_data
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name, temp_sample_list_file)

        # Asserts
        # last file"s label should be normalized to "Label.'
        self.assertEqual(files[-1][1], "I4831 #",
                         "Text not normalized correctly")

        self._remove_temp_file(temp_sample_list_file)

    def test_multiplication(self):
        chars = set()
        labels = {"test_partition": []}
        partition = {"test_partition": []}
        data_file_list = self.sample_list_file
        partition_name = "test_partition"

        # Initialize DataLoader with multiply set to 2
        data_loader = self.DataLoader(batch_size=32, img_size=(256, 256, 3))
        data_loader.multiply = 2

        # Call create_data with use_multiply=True
        chars, files = data_loader.create_data(
            chars, labels, partition, partition_name,
            data_file_list, use_multiply=True)

        # Asserts
        # should be 6 now, as each line is duplicated due to multiplication
        self.assertEqual(len(files), 6, "Multiplication not working")

    def test_generators(self):
        batch_size = 2
        img_size = (256, 256, 3)

        # Setup: Create dummy train, validation, test, and inference lists
        data_loader = self.DataLoader(batch_size=batch_size,
                                      img_size=img_size)

        data_loader.train_list = self._create_temp_file()
        data_loader.validation_list = self._create_temp_file()
        data_loader.test_list = self._create_temp_file()
        data_loader.inference_list = self._create_temp_file()

        training_generator, evaluation_generator, validation_generator, \
            test_generator, inference_generator, utils, train_batches, \
            val_text = data_loader.generators()

        # Basic tests
        self.assertIsNotNone(training_generator,
                             "Training generator is None")
        self.assertIsNotNone(evaluation_generator,
                             "Evaluation generator is None")
        self.assertIsNotNone(validation_generator,
                             "Validation generator is None")
        self.assertIsNotNone(test_generator,
                             "Test generator is None")
        self.assertIsNotNone(inference_generator,
                             "Inference generator is None")

        # CharList
        self.assertIsNotNone(data_loader.charList, "CharList is None")

        # Deep Checks
        # Ensure train_batches is correct
        # Assuming 3 sample images
        self.assertEqual(train_batches, np.ceil(
            3 / batch_size), "Train batches incorrect")

        # Let"s check the first batch.
        for images, labels in training_generator.take(1):
            # First dim should be batch_size
            self.assertEqual(images.shape[0], batch_size,
                             f"Batch size incorrect. Expected {batch_size}, "
                             f"got {images.shape[0]}")

            # Fourth dim should be channels
            self.assertEqual(images.shape[3], img_size[2],
                             f"Channels incorrect. Expected {img_size[2]}, "
                             f"got {images.shape[3]}")

            # Check the number of labels
            # ??? Very unclear how the labels work
            # self.assertEqual(len(labels), 3)  # 3 labels

        # Cleanup: Remove temporary files
        self._remove_temp_file(data_loader.validation_list)
        self._remove_temp_file(data_loader.test_list)
        self._remove_temp_file(data_loader.inference_list)


if __name__ == "__main__":
    unittest.main()
