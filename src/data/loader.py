# Imports

# > Standard library
import logging
import os
import re
import json


# > Third party dependencies
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np


# > Local dependencies
from data.generator import DataGenerator
from utils.text import Tokenizer


class DataLoader:
    """loader for dataset at given location, preprocess images and text
    according to parameters"""
    DTYPE = 'float32'
    currIdx = 0
    charList = []
    samples = []
    validation_dataset = []

    def __init__(self,
                 batch_size,
                 img_size,
                 char_list=[],
                 train_list='',
                 validation_list='',
                 test_list='',
                 inference_list='',
                 normalization_file=None,
                 multiply=1,
                 check_missing_files=True,
                 replace_final_layer=False,
                 use_mask=False,
                 augment_model=None
                 ):
        self.currIdx = 0
        self.batch_size = batch_size
        self.imgSize = img_size
        self.samples = []
        self.height = img_size[0]
        self.width = img_size[1]
        self.channels = img_size[2]
        self.partition = []
        self.injected_charlist = char_list
        self.train_list = train_list
        self.validation_list = validation_list
        self.test_list = test_list
        self.inference_list = inference_list
        self.normalization_file = normalization_file
        self.multiply = multiply
        self.check_missing_files = check_missing_files
        self.replace_final_layer = replace_final_layer
        self.use_mask = use_mask
        self.augment_model = augment_model

    @staticmethod
    def normalize(input: str, replacements: str) -> str:
        """
        Normalize text using a json file with replacements

        Parameters
        ----------
        input : str
            Input string to normalize
        replacements : str
            Path to json file with replacements, where key is the string to
            replace and value is the replacement. Example: {"a": "b"} will
            replace all "a" with "b" in the input string.

        Returns
        -------
        str
            Normalized string
        """

        with open(replacements, 'r') as f:
            replacements = json.load(f)
            for key, value in replacements.items():
                input = input.replace(key, value)

        input = re.sub(r"\s+", " ", input)

        return input.strip()

    def init_data_generator(self,
                            files,
                            params,
                            is_training=False,
                            deterministic=False):
        """
        Create DataGenerator object which is used to load and preprocess
        batches of data (tuples) consisting of a file path and a label.
        AUTOTUNE is applied to optimize parallel calls to the laod_images
        function.

        Parameters
        ----------
        files:
            Input files for data generator
        params: dict
            Dict of training parameters used to pre-process the data
        is_training: bool
            Indicate whether generator is used for training process or not
        deterministic: bool
            Control the order in which the transformation produces elements,
            If set to False, the transformation is allowed to yield elements
            out of order to trade determinism for performance.

        Returns
        ----------
        DataGenerator
            Generator object with loaded images and params
        """
        data_generator = DataGenerator(is_training=is_training, **params)

        num_batches = np.ceil(len(files) / self.batch_size)
        generator = tf.data.Dataset.from_tensor_slices(files)
        if is_training:
            # Add additional repeat and shuffle for training
            generator = generator.repeat().shuffle(len(files))

        generator = (generator
                     .map(data_generator.load_images,
                          num_parallel_calls=AUTOTUNE,
                          deterministic=deterministic)
                     .padded_batch(self.batch_size,
                                   padded_shapes=(
                                       [None, None, self.channels], [None]),
                                   padding_values=(
                                       tf.constant(-10, dtype=tf.float32),
                                       tf.constant(0, dtype=tf.int64)))
                     .prefetch(AUTOTUNE)
                     ).apply(
            tf.data.experimental.assert_cardinality(num_batches))

        return generator

    def get_generators(self):
        """
        Initializes data generators for different dataset partitions and updates
        character set and tokenizer based on the dataset.

        Firstly, data is created for

        """
        # Initialize character set and data partitions with corresponding labels.
        characters = set()
        partitions = {
            'train': [],
            'evaluation': [],
            'validation': [],
            'test': [],
            'inference': []
        }
        labels = {
            'train': [],
            'evaluation': [],
            'validation': [],
            'test': [],
            'inference': []
        }

        # Process training data and update characters set and partitions.
        if self.train_list:
            characters, train_files = self.create_data(
                characters, labels, partitions, 'train', self.train_list,
                use_multiply=True
            )

            # Process evaluation data if available.
            if self.validation_list:
                characters, eval_files = self.create_data(
                    characters, labels, partitions, 'evaluation',
                    self.validation_list
                )

        # TODO: Replace this by a do_validate flag
        # Process validation data if available.
        if self.validation_list:
            characters, val_files = self.create_data(
                characters, labels, partitions, 'validation',
                self.validation_list, include_unsupported_chars=True
            )

        # Process test data if available.
        if self.test_list:
            characters, test_files = self.create_data(
                characters, labels, partitions, 'test', self.test_list,
                include_unsupported_chars=True
            )

        # Process inference data if available.
        if self.inference_list:
            characters, inf_files = self.create_data(
                characters, labels, partitions, 'inference',
                self.inference_list, include_unsupported_chars=True,
                include_missing_files=True, is_inference=True
            )

        # Determine the character list for the tokenizer.
        if self.injected_charlist and not self.replace_final_layer:
            logging.info('Using injected charlist')
            self.charList = self.injected_charlist
        else:
            self.charList = sorted(list(characters))

        # Initialize the tokenizer.
        self.tokenizer = Tokenizer(self.charList, self.use_mask)

        # Define common parameters for all data generators.
        train_params = {
            'tokenizer': self.tokenizer,
            'height': self.height,
            'batch_size': self.batch_size,
            'channels': self.channels,
            'augment_model': self.augment_model
        }

        # Initialize data generators for each dataset partition as needed.
        training_generator = evaluation_generator = validation_generator = None
        test_generator = inference_generator = None
        train_batches = 0

        if self.train_list:
            training_generator = self.init_data_generator(
                train_files, train_params, is_training=True
            )
            train_batches = np.ceil(len(train_files) / self.batch_size)

        if self.validation_list:
            if self.train_list:
                evaluation_generator = self.init_data_generator(
                    eval_files, train_params, deterministic=True,
                    is_training=False
                )
            validation_generator = self.init_data_generator(
                val_files, train_params, deterministic=True,
                is_training=False
            )

        if self.test_list:
            test_generator = self.init_data_generator(
                test_files, train_params, deterministic=True,
                is_training=False
            )

        if self.inference_list:
            inference_generator = self.init_data_generator(
                inf_files, train_params, deterministic=True,
                is_training=False
            )

        # Update the partition information.
        self.partition = partitions

        # Return all initialized generators, tokenizer, and other relevant info.
        return (
            training_generator,
            evaluation_generator,
            validation_generator,
            test_generator,
            inference_generator,
            self.tokenizer,
            int(train_batches),
            labels['validation']
        )

    def create_data(self, characters, labels, partitions, partition_name,
                    data_files,
                    include_unsupported_chars=False,
                    include_missing_files=False,
                    is_inference=False, use_multiply=False):
        """
        Processes data files to create a dataset partition, updating characters,
        labels, and partition lists accordingly.

        Parameters:
        - characters: Set of characters found in the dataset.
        - labels: Dictionary mapping partition names to lists of labels.
        - partitions: Dictionary mapping partition names to lists of file paths.
        - partition_name: Name of the current partition being processed.
        - data_files: List of paths to data files.
        - include_unsupported_chars: Flag to include lines with unsupported characters.
        - include_missing_files: Flag to include missing files in the dataset.
        - is_inference: Flag to indicate processing for inference, where ground truth might be unknown.
        - use_multiply: Flag to duplicate entries based on the 'multiply' attribute for data augmentation.

        Returns:
        - Updated set of characters and list of processed files.
        """
        processed_files = []

        # Process each file in the data files list.
        for file_path in data_files.split():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist")

            with open(file_path) as file:
                valid_lines = 0  # Counter for valid lines.
                for line in file:
                    if not line or line[0] == '#':
                        continue  # Skip empty lines and comments.

                    line_parts = line.strip().split('\t')
                    if not is_inference and len(line_parts) == 1:
                        logging.warning(f"Empty ground truth in {line}. "
                                        f"Skipping for {partition_name}...")
                        continue

                    # filename
                    file_name = line_parts[0]
                    # Skip missing files unless explicitly included.
                    if not include_missing_files and self.check_missing_files \
                            and not os.path.exists(file_name):
                        logging.warning(f"Missing: {file_name} in {file_path}. "
                                        f"Skipping for {partition_name}...")
                        continue

                    # Determine the ground truth text.
                    if is_inference:
                        ground_truth = 'to be determined'
                    elif self.normalization_file and \
                            (partition_name == 'train'
                             or partition_name == 'evaluation'):
                        ground_truth = self.normalize(line_parts[1],
                                                      self.normalization_file)
                    else:
                        ground_truth = line_parts[1]

                    ignore_line = False

                    # We want to skip lines with unsupported characters, except
                    # for the training set, since we make our charlist from
                    # that
                    # If we're using an injected charlist, we want to skip
                    # unsupported characters in the training set as well
                    if not include_unsupported_chars \
                            and (partition_name != 'train'
                                 or self.injected_charlist):
                        for char in ground_truth:
                            if char not in self.injected_charlist and \
                                    char not in characters:
                                logging.warning("Unsupported character: "
                                                f"{char} in {ground_truth}. "
                                                "Skipping for "
                                                f"{partition_name}...")
                                ignore_line = True
                                break
                    if ignore_line or len(ground_truth) == 0:
                        continue
                    valid_lines += 1
                    if use_multiply:
                        # Multiply the data if requested
                        for _ in range(self.multiply):
                            partitions[partition_name].append(file_name)
                            labels[partition_name].append(ground_truth)
                            processed_files.append([file_name, ground_truth])
                    else:
                        # Or just combine file names and labels
                        partitions[partition_name].append(file_name)
                        labels[partition_name].append(ground_truth)
                        processed_files.append([file_name, ground_truth])
                    if (not self.injected_charlist or
                        self.replace_final_layer) \
                            and partition_name == 'train':
                        characters = characters.union(
                            set(char for label in ground_truth for char in label))

                logging.info(f"Found {valid_lines} lines suitable for "
                             f"{partition_name}")

        return characters, processed_files

    def get_item(self, partition, item_id):
        return self.partition[partition][item_id]
