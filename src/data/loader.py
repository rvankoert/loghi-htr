# Imports

# > Standard library
import logging

# > Local dependencies
from data.generator import DataGenerator
from utils.text import Tokenizer

# > Third party dependencies
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import numpy as np

# > Environment
import json
import re
import os


class DataLoader:
    DTYPE = 'float32'
    currIdx = 0
    charList = []
    samples = []
    validation_dataset = []

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

    def init_data_generator(self, files, params, is_training=False, deterministic=False):
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
                     ).apply(tf.data.experimental.assert_cardinality(num_batches))

        return generator

    def generators(self):
        chars = set()
        partition = {'train': [], 'evaluation': [], 'validation': [],
                     'test': [], 'inference': []}
        labels = {'train': [], 'evaluation': [], 'validation': [],
                  'test': [], 'inference': []}

        if self.train_list:
            chars, train_files = self.create_data(
                chars, labels, partition, 'train', self.train_list, use_multiply=True)

        if self.validation_list:
            if self.train_list:
                chars, evaluation_files = self.create_data(
                    chars, labels, partition, 'evaluation',
                    self.validation_list)

            chars, validation_files = self.create_data(
                chars, labels, partition, 'validation', self.validation_list,
                include_unsupported_chars=True)

        if self.test_list:
            chars, test_files = self.create_data(chars, labels, partition, 'test', self.test_list,
                                                 include_unsupported_chars=True)
        if self.inference_list:
            chars, inference_files = self.create_data(chars, labels, partition, 'inference', self.inference_list,
                                                      include_unsupported_chars=True, include_missing_files=True,
                                                      is_inference=True)

        # list of all chars in dataset
        if self.injected_charlist and not self.replace_final_layer:
            logging.info('Using injected charlist')
            self.charList = self.injected_charlist
        else:
            self.charList = sorted(list(chars))

        self.tokenizer = Tokenizer(self.charList, self.use_mask)

        # TODO: These are the same, so we can remove one of them
        train_params = {'tokenizer': self.tokenizer,
                        'height': self.height,
                        'batch_size': self.batch_size,
                        'channels': self.channels,
                        'augment_model': self.augment_model}
        non_train_params = {'tokenizer': self.tokenizer,
                            'batch_size': self.batch_size,
                            'height': self.height,
                            'channels': self.channels,
                            'augment_model': self.augment_model}

        training_generator = None
        evaluation_generator = None
        validation_generator = None
        test_generator = None
        inference_generator = None
        train_batches = 0
        if self.train_list:
            training_generator = self.init_data_generator(
                train_files, train_params, is_training=True)
            # Explicitly set train batches otherwise training is not initialised
            train_batches = np.ceil(len(train_files) / self.batch_size)
        if self.validation_list:
            if self.train_list:
                evaluation_generator = self.init_data_generator(
                    evaluation_files, non_train_params, deterministic=True,
                    is_training=True)
            validation_generator = self.init_data_generator(
                validation_files, non_train_params, deterministic=True)
        if self.test_list:
            test_generator = self.init_data_generator(
                test_files, non_train_params, deterministic=True)
        if self.inference_list:
            inference_generator = self.init_data_generator(
                inference_files, non_train_params, deterministic=True)

        self.partition = partition

        return training_generator, evaluation_generator, \
            validation_generator, test_generator, inference_generator, \
            self.tokenizer, int(train_batches), labels['validation']

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
        """loader for dataset at given location, preprocess images and text according to parameters"""
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

    def create_data(self, chars, labels, partition, partition_name, data_file_list, include_unsupported_chars=False,
                    include_missing_files=False, is_inference=False, use_multiply=False):
        files = []
        for sublist in data_file_list.split():
            if not os.path.exists(sublist):
                raise FileNotFoundError(f"{sublist} does not exist")
            with open(sublist) as f:
                counter = 0
                for line in f:
                    if not line or line[0] == '#':
                        continue
                    lineSplit = line.strip().split('\t')
                    if not is_inference and len(lineSplit) == 1:
                        logging.warning(f"Empty ground truth on {line}. "
                                        f"Skipping for {partition_name}...")
                        continue

                    # filename
                    fileName = lineSplit[0]
                    if not include_missing_files and self.check_missing_files \
                            and not os.path.exists(fileName):
                        logging.warning(f"Missing: {fileName} in {sublist}. "
                                        f"Skipping for {partition_name}...")
                        continue
                    if is_inference:
                        gtText = 'to be determined'

                    # Normalize text if normalization file is provided and
                    # we're training or evaluating
                    elif self.normalization_file and \
                            (partition_name == 'train'
                             or partition_name == 'evaluation'):
                        gtText = self.normalize(
                            lineSplit[1], self.normalization_file)
                    else:
                        gtText = lineSplit[1]
                    ignoreLine = False

                    # We want to skip lines with unsupported characters, except
                    # for the training set, since we make our charlist from
                    # that
                    # If we're using an injected charlist, we want to skip
                    # unsupported characters in the training set as well
                    if not include_unsupported_chars \
                            and (partition_name != 'train'
                                 or self.injected_charlist):
                        for char in gtText:
                            if char not in self.injected_charlist and \
                                    char not in chars:
                                logging.warning("Unsupported character: "
                                                f"{char} in {gtText}. "
                                                "Skipping for "
                                                f"{partition_name}...")
                                ignoreLine = True
                                break
                    if ignoreLine or len(gtText) == 0:
                        continue
                    counter = counter + 1
                    if use_multiply:
                        for i in range(0, self.multiply):
                            partition[partition_name].append(fileName)
                            labels[partition_name].append(gtText)
                            files.append([fileName, gtText])
                    else:
                        partition[partition_name].append(fileName)
                        labels[partition_name].append(gtText)
                        files.append([fileName, gtText])
                    if (not self.injected_charlist or
                            self.replace_final_layer) \
                            and partition_name == 'train':
                        chars = chars.union(
                            set(char for label in gtText for char in label))

                logging.info(f"Found {counter} lines suitable for "
                             f"{partition_name}")
        return chars, files

    def get_item(self, partition, item_id):
        return self.partition[partition][item_id]
