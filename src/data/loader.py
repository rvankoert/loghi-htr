# Imports

# > Standard library
from __future__ import division
from __future__ import print_function

# > Local dependencies
from data.generator import DataGenerator
from utils.utils import Utils

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
        data_generator = DataGenerator(**params)
        num_batches = np.ceil(len(files) / self.batch_size)
        generator = tf.data.Dataset.from_tensor_slices(files)

        if is_training:
            # Add additional repeat and shuffle for training
            generator = generator.repeat().shuffle(len(files))
        generator = (generator.map(data_generator.load_images,
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
        partition = {'train': [], 'validation': [],
                     'test': [], 'inference': []}
        labels = {'train': [], 'validation': [], 'test': [], 'inference': []}

        if self.train_list:
            chars, train_files = self.create_data(
                chars, labels, partition, 'train', self.train_list, use_multiply=True)

        if self.validation_list:
            chars, validation_files = self.create_data(
                chars, labels, partition, 'validation', self.validation_list)

        if self.test_list:
            chars, test_files = self.create_data(chars, labels, partition, 'test', self.test_list,
                                                 include_unsupported_chars=True)
        if self.inference_list:
            chars, inference_files = self.create_data(chars, labels, partition, 'inference', self.inference_list,
                                                      include_unsupported_chars=True, include_missing_files=True,
                                                      is_inference=True)

        # list of all chars in dataset
        if self.injected_charlist and not self.replace_final_layer:
            print('using injected charlist')
            self.charList = self.injected_charlist
        else:
            self.charList = sorted(list(chars))

        self.utils = Utils(self.charList, self.use_mask)

        train_params = {'utils': self.utils,
                        'height': self.height,
                        'batch_size': self.batch_size,
                        'channels': self.channels,
                        'aug_binarize_sauvola': self.aug_binarize_sauvola,
                        'aug_binarize_otsu': self.aug_binarize_otsu,
                        'aug_elastic_transform': self.aug_elastic_transform,
                        'aug_random_crop': self.aug_random_crop,
                        'aug_random_width': self.aug_random_width,
                        'aug_distort_jpeg': self.aug_distort_jpeg,
                        'aug_random_shear': self.aug_random_shear
                        }
        non_train_params = {'utils': self.utils,
                            'batch_size': self.batch_size,
                            'height': self.height,
                            'channels': self.channels,
                            'aug_binarize_sauvola': self.aug_binarize_sauvola,
                            'aug_binarize_otsu': self.aug_binarize_otsu
                            }

        training_generator = None
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
            validation_generator = self.init_data_generator(
                validation_files, non_train_params, deterministic=True)
        if self.test_list:
            test_generator = self.init_data_generator(
                test_files, non_train_params, deterministic=True)
        if self.inference_list:
            inference_generator = self.init_data_generator(
                inference_files, non_train_params, deterministic=True)

        self.partition = partition

        return training_generator, validation_generator, test_generator, inference_generator, self.utils, train_batches

    def __init__(self,
                 batch_size,
                 img_size,
                 char_list=None,
                 train_list='',
                 validation_list='',
                 test_list='',
                 inference_list='',
                 aug_binarize_sauvola=False,
                 aug_binarize_otsu=False,
                 normalization_file=None,
                 multiply=1,
                 augment=True,
                 aug_elastic_transform=False,
                 num_oov_indices=0,
                 aug_random_crop=False,
                 aug_random_width=False,
                 check_missing_files=True,
                 aug_distort_jpeg=False,
                 replace_final_layer=False,
                 use_mask=False,
                 aug_random_shear=False
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
        self.aug_binarize_sauvola = aug_binarize_sauvola
        self.aug_binarize_otsu = aug_binarize_otsu
        self.normalization_file = normalization_file
        self.multiply = multiply
        self.dataAugmentation = augment
        self.aug_elastic_transform = aug_elastic_transform
        self.num_oov_indices = num_oov_indices
        self.aug_random_crop = aug_random_crop
        self.aug_random_width = aug_random_width
        self.check_missing_files = check_missing_files
        self.aug_distort_jpeg = aug_distort_jpeg
        self.replace_final_layer = replace_final_layer
        self.use_mask = use_mask
        self.aug_random_shear = aug_random_shear

    def create_data(self, chars, labels, partition, partition_name, data_file_list, include_unsupported_chars=False,
                    include_missing_files=False, is_inference=False, use_multiply=False):
        files = []
        for sublist in data_file_list.split():
            if not os.path.exists(sublist):
                print(sublist + "does not exist, enter a valid filename. exiting...")
                exit(1)
            with open(sublist) as f:
                counter = 0
                for line in f:
                    if not line or line[0] == '#':
                        continue
                    lineSplit = line.strip().split('\t')
                    if not is_inference and len(lineSplit) == 1:
                        continue

                    # filename
                    fileName = lineSplit[0]
                    if not include_missing_files and self.check_missing_files and not os.path.exists(fileName):
                        print("missing: " + fileName)
                        continue
                    if is_inference:
                        gtText = 'to be determined'
                    elif self.normalization_file:
                        gtText = self.normalize(
                            lineSplit[1], self.normalization_file)
                    else:
                        gtText = lineSplit[1]
                    ignoreLine = False
                    if not include_unsupported_chars and self.injected_charlist and not self.replace_final_layer:
                        for char in gtText:
                            if char not in self.injected_charlist:
                                print('a ignoring line: ' + gtText)
                                ignoreLine = True
                                break
                    if ignoreLine or len(gtText) == 0:
                        print(line)
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
                    if not self.injected_charlist or self.replace_final_layer:
                        chars = chars.union(
                            set(char for label in gtText for char in label))
                print('found ' + str(counter) +
                      ' lines suitable for ' + partition_name)
        return chars, files

    @staticmethod
    def truncate_label(text, max_text_len):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def get_item(self, partition, item_id):
        return self.partition[partition][item_id]
