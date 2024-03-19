# Imports

# > Standard library
from collections import defaultdict
import logging
import os
from typing import Dict, Tuple, Optional, List, Set

# > Third party dependencies
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np

# > Local dependencies
from data.loader import DataLoader
from setup.config import Config
from utils.text import Tokenizer, normalize_text


class DataManager:
    """
    Class for creating and managing datasets for training, validating, etc.

    Parameters
    ----------
    img_size : Tuple[int, int, int]
        The size of the input images (height, width, channels).
    augment_model : tf.keras.Sequential
        The model used for data augmentation.
    config : Config
        The configuration dictionary containing various settings.
    charlist : Optional[List[str]], optional
        The list of characters to use for tokenization, by default None.
    """

    def __init__(self,
                 img_size: Tuple[int, int, int],
                 augment_model: tf.keras.Sequential,
                 config: Config,
                 charlist: Optional[List[str]] = None,
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

        # Determine the evaluation list
        self.evaluation_list = None
        if self.config['train_list'] and self.config['validation_list']:
            self.evaluation_list = self.config['validation_list']

        # Remove the evaluation list from the validation list if we are not
        # doing validation
        if not self.config['do_validate']:
            self.config['validation_list'] = ""

        # Process the raw data and create file names, labels, sample weights,
        # and tokenizer
        logging.info("Processing raw data...")
        file_names, labels, sample_weights, self.tokenizer \
            = self._process_raw_data()

        self.raw_data = {split: (file_names[split], labels[split],
                                 sample_weights[split])
                         for split in ['train', 'evaluation', 'validation',
                                       'test', 'inference']}

        # Fill the datasets dictionary with datasets for different partitions
        logging.info("Creating datasets...")
        self.datasets = self._fill_datasets_dict(file_names, labels,
                                                 sample_weights)

    def _process_raw_data(self) -> Tuple[Dict[str, List[str]],
                                         Dict[str, List[str]],
                                         Dict[str, List[str]],
                                         Tokenizer]:
        """
        Process the raw data and create file names, labels, sample weights,
        and tokenizer.

        Returns
        -------
        Tuple[Dict[str, List[str]],
              Dict[str, List[str]],
              Dict[str, List[str]],
              Tokenizer]
            A tuple containing dictionaries of file names, labels, sample
            weights, and the tokenizer.
        """

        # Initialize character set and data partitions with corresponding
        # labels
        file_names_dict = defaultdict(list)
        labels_dict = defaultdict(list)
        sample_weights_dict = defaultdict(list)

        for partition in ('train', 'evaluation', 'validation',
                          'test', 'inference'):
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

    def _fill_datasets_dict(self,
                            partitions: Dict[str, List[str]],
                            labels: Dict[str, List[str]],
                            sample_weights: Dict[str, List[str]]) \
            -> Dict[str, tf.data.Dataset]:
        """
        Initialize data generators for different dataset partitions and
        update character set and tokenizer based on the dataset.

        Parameters
        ----------
        partitions : Dict[str, List[str]]
            A dictionary containing lists of file names for each partition.
        labels : Dict[str, List[str]]
            A dictionary containing lists of labels for each partition.
        sample_weights : Dict[str, List[str]]
            A dictionary containing lists of sample weights for each partition.

        Returns
        -------
        Dict[str, tf.data.Dataset]
            A dictionary containing datasets for each partition.
        """

        # Create datasets for different partitions
        datasets = defaultdict(lambda: None)

        for partition in ('train', 'evaluation', 'validation',
                          'test', 'inference'):
            # Special case for evaluation partition, since there is no
            # evaluation_list in the config, but the evaluation_list is
            # inferred from the validation_list and train_list in the init
            if partition == "evaluation":
                partition_list = self.evaluation_list
            else:
                partition_list = self.config[f"{partition}_list"]
            if partition_list:
                # Create dataset for the current partition
                datasets[partition] = self._create_dataset(
                    files=partitions[partition],
                    labels=labels[partition],
                    sample_weights=sample_weights[partition],
                    partition_name=partition
                )

        return datasets

    def _create_data(self,
                     partition_name: str,
                     text_file: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Create data for a specific partition from a text file.

        Parameters
        ----------
        partition_name : str
            The name of the partition.
        text_file : str
            The path to the text file containing the data.

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            A tuple containing lists of file names, labels, and sample weights.
        """

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
            logging.debug("Updated charlist: %s", self.charlist)

        logging.info("Created data for %s with %s samples",
                     partition_name, len(partitions))

        return partitions, labels, sample_weights

    def _process_line(self,
                      line: str,
                      partition_name: str,
                      characters: Set[str]) \
        -> Tuple[Optional[Tuple[str, str, float]],
                 Optional[str]]:
        """
        Process a single line from the data file.

        Parameters
        ----------
        line : str
            The line to process.
        partition_name : str
            The name of the partition.
        characters : Set[str]
            The set of characters to use for validation.

        Returns
        -------
        Tuple[Optional[Tuple[str, str, float]], Optional[str]]
            A tuple containing the processed data (file name, ground truth,
            sample weight) and the flaw (if any).
        """

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
                partition_name in ('train' or 'evaluation'):
            ground_truth = normalize_text(ground_truth,
                                          self.config['normalization_file'])

        # Check for unsupported characters in the ground truth
        if not self._is_valid_ground_truth(ground_truth, partition_name,
                                           characters):
            return None, "Unsupported characters"

        sample_weight = self._get_sample_weight(fields)

        return (file_name, ground_truth, sample_weight), None

    def _get_ground_truth(self,
                          fields: List[str],
                          partition_name: str) \
            -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the ground truth from the fields.

        Parameters
        ----------
        fields : List[str]
            The fields from the line.
        partition_name : str
            The name of the partition.

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            A tuple containing the ground truth and the flaw (if any).
        """

        # Collect the ground truth and skip lines with empty ground truth
        # unless it's an inference partition
        ground_truth = fields[-1] if len(fields) > 1 else ""

        if not ground_truth:
            if partition_name == "inference":
                ground_truth = "to be determined"
            else:
                return None, "Empty ground truth"

        return ground_truth, None

    def _is_valid_ground_truth(self,
                               ground_truth: str,
                               partition_name: str,
                               characters: Set[str]) -> bool:
        """
        Check if the ground truth is valid.

        Parameters
        ----------
        ground_truth : str
            The ground truth to check.
        partition_name : str
            The name of the partition.
        characters : Set[str]
            The set of characters.

        Returns
        -------
        bool
            True if the ground truth is valid, False otherwise.
        """

        # Check for unsupported characters in the ground truth
        # and update the character set if the partition is 'train'
        unsupported_characters = set(ground_truth) - characters
        if unsupported_characters:
            # Unsupported characters are allowed in the validation, inference,
            # and test partitions, but not in the evaluation partition
            if partition_name in ('validation', 'inference', 'test'):
                return True

            if partition_name == 'train' and not self.charlist:
                characters.update(unsupported_characters)
                return True
            return False
        return True

    def _get_sample_weight(self,
                           fields: List[str]) -> float:
        """
        Extract the sample weight from the fields.

        Parameters
        ----------
        fields : List[str]
            The fields from the line.

        Returns
        -------
        float
            The sample weight.
        """

        # Extract the sample weight from the fields
        sample_weight = 1.0
        if len(fields) > 2:
            try:
                sample_weight = float(fields[1])
            except ValueError:
                pass
        return sample_weight

    def get_filename(self, partition: str, item_id: int):
        """ Get the filename for the given partition and item id """
        return self.raw_data[partition][0][item_id]

    def get_ground_truth(self, partition: str, item_id: int):
        """ Get the ground truth for the given partition and item id """
        return self.raw_data[partition][1][item_id]

    def get_train_batches(self):
        """ Get the number of batches for training """
        return int(np.ceil(len(self.raw_data['train'][0])
                           / self.config['batch_size']))

    def _create_dataset(self,
                        files: List[str],
                        labels: List[str],
                        sample_weights: List[str],
                        partition_name: str) -> tf.data.Dataset:
        """
        Create a dataset for a specific partition.

        Parameters
        ----------
        files : List[str]
            A list of file names.
        labels : List[str]
            A list of labels.
        sample_weights : List[str]
            A list of sample weights.
        partition_name : str
            The name of the partition.

        Returns
        -------
        tf.data.Dataset
            The created dataset.
        """

        # Determine if the dataset is for training
        is_training = partition_name == 'train'

        # Create the data loader that will be used to load and preprocess the
        # images in the dataset
        data_loader = DataLoader(self.tokenizer, self.augment_model,
                                 self.height, self.channels, is_training)

        # Zip the files, labels, and sample weights to create the dataset
        data = list(zip(files, labels, sample_weights))

        # Determine the number of batches
        num_batches = np.ceil(len(data) / self.config["batch_size"])

        # Create the dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if is_training:
            # Add additional repeat and shuffle for training
            dataset = dataset.repeat().shuffle(len(files))

        dataset = (dataset
                   .map(data_loader.load_images,
                        num_parallel_calls=AUTOTUNE,
                        deterministic=not is_training)
                   .padded_batch(self.config["batch_size"],
                                 padded_shapes=(
                                     [None, None, self.channels], [None], []),
                                 padding_values=(
                                     tf.constant(-10, dtype=tf.float32),
                                     tf.constant(0, dtype=tf.int64),
                                     tf.constant(1.0, dtype=tf.float32)))
                   .prefetch(AUTOTUNE))\
            .apply(tf.data.experimental.assert_cardinality(num_batches))

        return dataset
