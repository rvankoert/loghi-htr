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

        # Determine the character list
        if charlist and not config['replace_final_layer']:
            logging.info('Using injected charlist')
            self.charlist = sorted(list(charlist))
        else:
            self.charlist = []

        self.evaluation_list = None
        if self.config['train_list'] and self.config['validation_list']:
            self.evaluation_list = self.config['validation_list']

        if not self.config['do_validate']:
            self.config['validation_list'] = ""

        file_names, labels, sample_weights, self.tokenizer \
            = self._process_raw_data()
        self.raw_data = {split: (file_names[split], labels[split],
                                 sample_weights[split])
                         for split in ['train', 'evaluation', 'validation',
                                       'test', 'inference']}

        self.datasets = self._fill_datasets_dict(
            file_names, labels, sample_weights)

    def _process_raw_data(self):
        # Initialize character set and data partitions with corresponding
        # labels
        file_names_dict = defaultdict(list)
        labels_dict = defaultdict(list)
        sample_weights_dict = defaultdict(list)

        for partition in ['train', 'evaluation', 'validation',
                          'test', 'inference']:
            partition_text_file = self.config[f"{partition}_list"] \
                if partition != "evaluation" else self.evaluation_list

            if partition_text_file:
                # Create data for the current partition
                file_names, labels, sample_weights = self._create_data(
                    partition_name=partition,
                    text_file=partition_text_file,
                )

                # Fill the dictionary with the data
                file_names_dict[partition] = file_names
                labels_dict[partition] = labels
                sample_weights_dict[partition] = sample_weights

        # Initialize the tokenizer
        tokenizer = Tokenizer(self.charlist, self.config['use_mask'])

        return file_names_dict, labels_dict, sample_weights_dict, tokenizer

    def _fill_datasets_dict(self, partitions, labels, sample_weights):
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
                files = [(partition, label, sample_weight)
                         for partition, label, sample_weight in
                         zip(partitions[partition], labels[partition],
                             sample_weights[partition])]
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

    def _create_data(self, partition_name, text_file):
        # Define the lists for the current partition
        labels, partitions, sample_weights = [], [], []

        # Define the faulty lines and flaws
        faulty_lines = {}
        flaw_counts = {}

        # Define the character set
        characters = set(self.charlist)

        # Process each file in the data files list
        for file_path in text_file.split():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist")

            with open(file_path, encoding="utf-8") as file:
                # Iterate over the lines in the file
                for line in file:
                    data, flaw = self._process_line(line,
                                                    partition_name,
                                                    characters)
                    if data is not None:
                        file_name, ground_truth, sample_weight = data
                        partitions.append(file_name)
                        labels.append(ground_truth)
                        sample_weights.append(str(sample_weight))
                    else:
                        faulty_lines[line] = flaw
                        flaw_counts[flaw] = flaw_counts.get(flaw, 0) + 1

        # Log the faulty lines and flaw counts
        if faulty_lines:
            logging.warning("Faulty lines for %s:", partition_name)
            for line, flaw in faulty_lines.items():
                logging.warning("%s: %s", line.strip(), flaw)
            logging.warning("Flaw counts for %s:", partition_name)
            for flaw, count in flaw_counts.items():
                logging.warning("%s: %d", flaw, count)

        # Update the charlist if it has changed
        if not self.charlist:
            self.charlist = sorted(list(characters))
            logging.info("Created charlist: %s", self.charlist)

        logging.info("Created data for %s with %s samples",
                     partition_name, len(partitions))

        return partitions, labels, sample_weights

    def _process_line(self, line, partition_name, characters):
        # Strip the line of leading and trailing whitespace
        line = line.strip()

        # Skip empty and commented lines
        if not line or line.startswith('#'):
            return None, "Empty or commented line"

        # Split the line into tab-separated fields:
        # filename sample_weight ground_truth
        fields = line.split('\t')

        # Extract the filename and ground truth from the fields
        file_name = fields[0]

        # Skip missing files
        if not os.path.exists(file_name):
            logging.warning("Missing: %s in %s. Skipping...",
                            file_name, partition_name)
            return None, "Missing file"

        # Extract the ground truth from the fields
        ground_truth, flaw = self._get_ground_truth(fields, partition_name)
        if ground_truth is None:
            return None, flaw

        # Normalize the ground truth if a normalization file is provided and
        # the partition is either 'train' or 'evaluation'
        if self.config['normalization_file'] and \
                (partition_name == 'train' or partition_name == 'evaluation'):
            ground_truth = normalize_text(ground_truth,
                                          self.config['normalization_file'])

        # Check for unsupported characters in the ground truth
        if not self._is_valid_ground_truth(ground_truth, partition_name,
                                           characters):
            return None, "Unsupported characters"

        sample_weight = self._get_sample_weight(fields)

        return (file_name, ground_truth, sample_weight), None

    def _get_ground_truth(self, fields, partition_name):
        # Collect the ground truth and skip lines with empty ground truth
        # unless it's an inference partition
        ground_truth = fields[-1] if len(fields) > 1 else ""

        if not ground_truth:
            if partition_name == "inference":
                ground_truth = "INFERENCE"
            else:
                return None, "Empty ground truth"

        return ground_truth, None

    def _is_valid_ground_truth(self, ground_truth, partition_name, characters):
        # Check for unsupported characters in the ground truth
        # Evaluation partition is allowed to have unsupported characters for a
        # more realistic evaluation
        unsupported_characters = set(ground_truth) - characters
        if unsupported_characters:
            if partition_name in ['validation', 'inference', 'test']:
                return True
            elif partition_name == 'train' and not self.charlist:
                characters.update(unsupported_characters)
                return True
            else:
                return False
        return True

    def _get_sample_weight(self, fields):
        # Extract the sample weight from the fields
        sample_weight = 1.0
        if len(fields) > 2:
            try:
                sample_weight = float(fields[1])
            except ValueError:
                pass
        return sample_weight

    def get_filename(self, partition, item_id):
        return self.raw_data[partition][0][item_id]

    def get_ground_truth(self, partition, item_id):
        return self.raw_data[partition][1][item_id]

    def get_train_batches(self):
        return int(np.ceil(len(self.raw_data['train'])
                           / self.config['batch_size']))


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
        # TODO: Use config["aug_multiply"] to multiply the data
        generator = generator.repeat().shuffle(len(files))

    generator = (generator
                 .map(data_generator.load_images,
                      num_parallel_calls=AUTOTUNE,
                      deterministic=deterministic)
                 .padded_batch(batch_size,
                               padded_shapes=(
                                   [None, None, channels], [None], []),
                               padding_values=(
                                   tf.constant(-10, dtype=tf.float32),
                                   tf.constant(0, dtype=tf.int64),
                                   tf.constant(1.0, dtype=tf.float32)))
                 .prefetch(AUTOTUNE)
                 ).apply(
        tf.data.experimental.assert_cardinality(num_batches))

    return generator
