# Imports

# > Standard library
from collections import defaultdict
import logging
import os

# > Third party dependencies
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np

# > Local dependencies
from data.generator import DataGenerator
from utils.text import Tokenizer, normalize_text


class DataLoader:
    """loader for dataset at given location, preprocess images and text
    according to parameters"""

    def __init__(self,
                 img_size,
                 augment_model,
                 config,
                 charlist=None,
                 ):

        self.augment_model = augment_model
        self.height = img_size[0]
        self.channels = img_size[2]
        self.config = config

        # TODO: Make this more clear
        self.charlist = charlist

        self.evaluation_list = None
        if self.config['do_validate']:
            self.evaluation_list = self.config['validation_list']

        partitions, labels, self.tokenizer = self._process_raw_data()
        self.raw_data = {split: (partitions[split], labels[split])
                         for split in ['train', 'evaluation', 'validation',
                                       'test', 'inference']}

        self.datasets = self._fill_datasets_dict(partitions, labels)

    def _process_raw_data(self):
        # Initialize character set and data partitions with corresponding
        # labels
        characters = set()
        partitions = defaultdict(list)
        labels = defaultdict(list)

        for partition in ['train', 'evaluation', 'validation',
                          'test', 'inference']:
            if partition == "evaluation":
                partition_list = self.evaluation_list
            else:
                partition_list = self.config[f"{partition}_list"]
            if partition_list:
                include_unsupported_chars = partition in ['validation', 'test']
                use_multiply = partition == 'train'
                characters, _ = self.create_data(
                    characters=characters,
                    labels=labels,
                    partitions=partitions,
                    partition_name=partition,
                    data_files=partition_list,
                    use_multiply=use_multiply,
                    include_unsupported_chars=include_unsupported_chars
                )

        # Determine the character list for the tokenizer
        if self.charlist and not self.config['replace_final_layer']:
            logging.info('Using injected charlist')
        else:
            self.charlist = sorted(list(characters))

        # Initialize the tokenizer
        tokenizer = Tokenizer(self.charlist, self.config['use_mask'])

        return partitions, labels, tokenizer

    def _fill_datasets_dict(self, partitions, labels):
        """
        Initializes data generators for different dataset partitions and
        updates character set and tokenizer based on the dataset.
        """

        # Create datasets for different partitions
        datasets = defaultdict(lambda: None)

        for partition in ['train', 'evaluation', 'validation',
                          'test', 'inference']:
            if partition == "evaluation":
                partition_list = self.evaluation_list
            else:
                partition_list = self.config[f"{partition}_list"]
            if partition_list:
                # Create dataset for the current partition
                files = list(zip(partitions[partition], labels[partition]))
                datasets[partition] = create_dataset(
                    files=files,
                    tokenizer=self.tokenizer,
                    augment_model=self.augment_model,
                    height=self.height,
                    channels=self.channels,
                    batch_size=self.config['batch_size'],
                    is_training=partition == 'train',
                    deterministic=partition != 'train'
                )

        return datasets

    def get_train_batches(self):
        return int(np.ceil(len(self.raw_data['train'])
                           / self.config['batch_size']))

    def create_data(self, characters, labels, partitions, partition_name,
                    data_files,
                    include_unsupported_chars=False,
                    include_missing_files=False,
                    is_inference=False,
                    use_multiply=False):
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
                    if not include_missing_files and \
                            self.config['check_missing_files'] and \
                            not os.path.exists(file_name):
                        logging.warning(f"Missing: {file_name} in {file_path}. "
                                        f"Skipping for {partition_name}...")
                        continue

                    # Determine the ground truth text.
                    if is_inference:
                        ground_truth = 'to be determined'
                    elif self.config['normalization_file'] and \
                            (partition_name == 'train'
                             or partition_name == 'evaluation'):
                        ground_truth = normalize_text(
                            line_parts[1],
                            self.config['normalization_file'])
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
                                 or self.charlist):
                        for char in ground_truth:
                            if char not in self.charlist and \
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
                        for _ in range(self.config['aug_multiply']):
                            partitions[partition_name].append(file_name)
                            labels[partition_name].append(ground_truth)
                            processed_files.append([file_name, ground_truth])
                    else:
                        # Or just combine file names and labels
                        partitions[partition_name].append(file_name)
                        labels[partition_name].append(ground_truth)
                        processed_files.append([file_name, ground_truth])
                    if (not self.charlist or
                        self.config['replace_final_layer']) \
                            and partition_name == 'train':
                        characters = characters.union(
                            set(char for label in ground_truth for char in label))

                logging.info(f"Found {valid_lines} lines suitable for "
                             f"{partition_name}")

        return characters, processed_files

    def get_filename(self, partition, item_id):
        return self.raw_data[partition][0][item_id]

    def get_ground_truth(self, partition, item_id):
        return self.raw_data[partition][1][item_id]


def create_dataset(files,
                   tokenizer,
                   augment_model,
                   height,
                   channels,
                   batch_size,
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

    data_generator = DataGenerator(tokenizer, augment_model, height,
                                   channels, is_training)

    num_batches = np.ceil(len(files) / batch_size)
    generator = tf.data.Dataset.from_tensor_slices(files)
    if is_training:
        # Add additional repeat and shuffle for training
        generator = generator.repeat().shuffle(len(files))

    generator = (generator
                 .map(data_generator.load_images,
                      num_parallel_calls=AUTOTUNE,
                      deterministic=deterministic)
                 .padded_batch(batch_size,
                               padded_shapes=([None, None, channels], [None]),
                               padding_values=(
                                   tf.constant(-10, dtype=tf.float32),
                                   tf.constant(0, dtype=tf.int64)))
                 .prefetch(AUTOTUNE)
                 ).apply(
        tf.data.experimental.assert_cardinality(num_batches))

    return generator
