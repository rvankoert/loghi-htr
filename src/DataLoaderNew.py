from __future__ import division
from __future__ import print_function

import os
import random

from tensorflow.python.data.experimental import AutoShardPolicy
import tensorflow as tf
from tensorflow.data import AUTOTUNE

from DataGeneratorLmdb import DataGeneratorLmdb
from DataGeneratorNew import DataGeneratorNew
from DataGeneratorNew2 import DataGeneratorNew2
import numpy as np
from utils import Utils


class DataLoaderNew:
    DTYPE = 'float32'

    currIdx = 0
    charList = []
    samples = []
    validation_dataset = []

    @staticmethod
    def normalize(input):
        output = input.replace(',,', '„') \
            .replace(' ,', ',') \
            .replace(',', ', ') \
             .replace(' .', '.') \
            .replace('.', '. ') \
            .replace('  ', ' ') \
            .replace('`', '\'') \
            .replace('´', '\'') \
            .replace('ʼ', '\'') \
            .replace('‘', '\'') \
            .replace('’', '\'') \
            .replace('“', '"') \
            .replace('”', '"') \
            .replace('·', '.') \
            .strip()
        return output

    def generators(self):
        chars = set()
        partition = {'train': [], 'validation': [], 'test': [], 'inference': []}
        labels = {'train': [], 'validation': [], 'test': [], 'inference': []}

        if self.train_list:
            chars, train_files = self.create_data(chars, labels, partition, 'train', self.train_list)

        if self.validation_list:
            chars, validation_files = self.create_data(chars, labels, partition, 'validation', self.validation_list)

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

        trainParams = {'height': self.height,
                       'shuffle': True,
                       'batch_size': self.batchSize,
                       'channels': self.channels,
                       'do_binarize_sauvola': self.do_binarize_sauvola,
                       'do_binarize_otsu': self.do_binarize_otsu,
                       'augment': self.dataAugmentation,
                       'do_elastic_transform': self.elastic_transform,
                       'random_crop': self.random_crop,
                       'random_width': self.random_width,
                       'distort_jpeg': self.distort_jpeg
                       }
        validationParams = {'shuffle': False,
                            'batch_size': self.batchSize,
                            'height': self.height,
                            'channels': self.channels,
                            'do_binarize_sauvola': self.do_binarize_sauvola,
                            'do_binarize_otsu': self.do_binarize_otsu,
                            }
        testParams = {'shuffle': False,
                      'batch_size': self.batchSize,
                      'height': self.height,
                      'channels': self.channels,
                      'do_binarize_sauvola': self.do_binarize_sauvola,
                      'do_binarize_otsu': self.do_binarize_otsu,
                      }
        inference_params = {'shuffle': False,
                            'batch_size': self.batchSize,
                            'height': self.height,
                            'channels': self.channels,
                            'do_binarize_sauvola': self.do_binarize_sauvola,
                            'do_binarize_otsu': self.do_binarize_otsu,
                            }
        training_generator = None
        validation_generator = None
        test_generator = None
        inference_generator = None
        use_classic = False
        deterministic = False
        train_batches=0
        if self.train_list:
            if use_classic:
                training_generator = self.create_data_generator(labels,
                                                                partition,
                                                                trainParams,
                                                                'train',
                                                                reuse_old_lmdb=self.reuse_old_lmdb_train)
            else:
                data_generator_new2 = DataGeneratorNew2(self.utils, 
                                                        self.batchSize,
                                                        channels=self.channels,
                                                        do_binarize_sauvola=self.do_binarize_sauvola,
                                                        do_binarize_otsu=self.do_binarize_otsu,
                                                        # augment=self.au,
                                                        do_elastic_transform=self.elastic_transform,
                                                        random_crop=self.random_crop,
                                                        random_width=self.random_width,
                                                        distort_jpeg=self.distort_jpeg
                                                        )
                train_batches = np.ceil(len(train_files) / self.batchSize)
                training_generator = tf.data.Dataset.from_tensor_slices(train_files)
                training_generator = (training_generator
                                      .repeat()
                                      .shuffle(len(train_files))
                                      .map(data_generator_new2.load_images, num_parallel_calls=AUTOTUNE,
                                           deterministic=deterministic)
                                      .padded_batch(self.batchSize, padded_shapes=([None, None, self.channels], [None]),
                                                    padding_values=(
                                                        tf.constant(-10, dtype=tf.float32),
                                                        tf.constant(0, dtype=tf.int64))
                                                    )
                                      .prefetch(AUTOTUNE)
                                      ).apply(tf.data.experimental.assert_cardinality(train_batches))

        if self.validation_list:
            if use_classic:
                validation_generator = self.create_data_generator(labels,
                                                                  partition,
                                                                  validationParams,
                                                                  'validation',
                                                                  reuse_old_lmdb=self.reuse_old_lmdb_val
                                                                  )
            else:
                data_generator_new2 = DataGeneratorNew2(self.utils,
                                                        self.batchSize,
                                                        height=self.height,
                                                        channels=self.channels,
                                                        do_binarize_sauvola=self.do_binarize_sauvola,
                                                        do_binarize_otsu=self.do_binarize_otsu,
                                                        )
                num_batches = np.ceil(len(validation_files) / self.batchSize)
                print('validation batches: ' + str(num_batches))
                validation_generator = tf.data.Dataset.from_tensor_slices(validation_files)
                validation_generator = (validation_generator
                                        # .repeat()
                                        .shuffle(len(validation_files))
                                        .map(data_generator_new2.load_images, num_parallel_calls=AUTOTUNE,
                                             deterministic=deterministic)
                                        .padded_batch(self.batchSize,
                                                      padded_shapes=([None, None, self.channels], [None]),
                                                      padding_values=(
                                                          tf.constant(-10, dtype=tf.float32),
                                                          tf.constant(0, dtype=tf.int64))
                                                      )
                                        .prefetch(AUTOTUNE)
                                        ).apply(tf.data.experimental.assert_cardinality(num_batches))
        if self.test_list:
            if use_classic:
                test_generator = self.create_data_generator(labels,
                                                            partition,
                                                            testParams,
                                                            'test',
                                                            reuse_old_lmdb=self.reuse_old_lmdb_test
                                                            )
            else:
                data_generator_new2 = DataGeneratorNew2(self.utils,
                                                        self.batchSize,
                                                        height=self.height,
                                                        channels=self.channels,
                                                        do_binarize_sauvola=self.do_binarize_sauvola,
                                                        do_binarize_otsu=self.do_binarize_otsu,)
                num_batches = np.ceil(len(test_files) / self.batchSize)
                test_generator = tf.data.Dataset.from_tensor_slices(test_files)
                test_generator = (test_generator
                                  .repeat()
                                  .shuffle(len(test_files))
                                  .map(data_generator_new2.load_images, num_parallel_calls=AUTOTUNE,
                                       deterministic=deterministic)
                                  .padded_batch(self.batchSize, padded_shapes=([None, None, self.channels], [None]),
                                                padding_values=(
                                                    tf.constant(-10, dtype=tf.float32), tf.constant(0, dtype=tf.int64))
                                                )
                                  .prefetch(AUTOTUNE)
                                  ).apply(tf.data.experimental.assert_cardinality(num_batches))

        if self.inference_list:
            # if use_classic:
            inference_generator = self.create_data_generator(labels,
                                                             partition,
                                                             inference_params,
                                                             'inference',
                                                             reuse_old_lmdb=self.reuse_old_lmdb_inference
                                                             )
            # else:
            #     dataGeneratorNew2 = DataGeneratorNew2(self.utils,
            #                                                         self.batchSize,
            #                                                         channels=self.channels,
            #                                                         do_binarize_sauvola=self.do_binarize_sauvola,
            #                                                         do_binarize_otsu=self.do_binarize_otsu)
            #     num_batches = np.ceil(len(inference_files) / self.batchSize)
            #     inference_generator = tf.data.Dataset.from_tensor_slices(inference_files)
            #     inference_generator = (inference_generator
            #                            .repeat()
            #                            .shuffle(len(inference_files))
            #                            .map(dataGeneratorNew2.load_images,
            #                                 num_parallel_calls=AUTOTUNE,
            #                                 deterministic=deterministic)
            #                            .padded_batch(self.batchSize,
            #                                          padded_shapes=([None, None, self.channels], [None]),
            #                                          padding_values=(
            #                                              tf.constant(-10, dtype=tf.float32),
            #                                              tf.constant(0, dtype=tf.int64)
            #                                          )
            #                                          )
            #                            .prefetch(AUTOTUNE)
            #                            ).apply(tf.data.experimental.assert_cardinality(num_batches))
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
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 normalize_text=False,
                 multiply=1,
                 augment=True,
                 elastic_transform=False,
                 num_oov_indices=0,
                 random_crop=False,
                 random_width=False,
                 check_missing_files=True,
                 distort_jpeg=False,
                 replace_final_layer=False,
                 use_lmdb=False,
                 reuse_old_lmdb_train=None,
                 reuse_old_lmdb_val=None,
                 reuse_old_lmdb_test=None,
                 reuse_old_lmdb_inference=None,
                 use_mask=False
                 ):
        """loader for dataset at given location, preprocess images and text according to parameters"""

        self.currIdx = 0
        self.batchSize = batch_size
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
        self.do_binarize_sauvola = do_binarize_sauvola
        self.do_binarize_otsu = do_binarize_otsu
        self.normalize_text = normalize_text
        self.multiply = multiply
        self.dataAugmentation = augment
        self.elastic_transform = elastic_transform
        self.num_oov_indices = num_oov_indices
        self.random_crop = random_crop
        self.random_width = random_width
        self.check_missing_files = check_missing_files
        self.distort_jpeg = distort_jpeg
        self.replace_final_layer = replace_final_layer
        self.use_lmdb = use_lmdb
        self.reuse_old_lmdb_train = reuse_old_lmdb_train
        self.reuse_old_lmdb_val = reuse_old_lmdb_val
        self.reuse_old_lmdb_test = reuse_old_lmdb_test
        self.reuse_old_lmdb_inference = reuse_old_lmdb_inference
        self.use_mask = use_mask

    def create_data_generator(self, labels, partition, params, process_step, reuse_old_lmdb=None):
        if self.use_lmdb:
            return DataGeneratorLmdb(
                partition[process_step],
                labels[process_step],
                **params,
                charList=self.charList,
                num_oov_indices=self.num_oov_indices,
                reuse_old_lmdb=reuse_old_lmdb
            )
        else:
            return DataGeneratorNew(
                partition[process_step],
                labels[process_step],
                **params,
                charList=self.charList,
                num_oov_indices=self.num_oov_indices
            )

    def create_data(self, chars, labels, partition, partition_name, data_file_list, include_unsupported_chars=False,
                    include_missing_files=False, is_inference=False):
        files = []
        for sublist in data_file_list.split():
            if not os.path.exists(sublist):
                print(sublist + "does not exist, enter a valid filename. exiting...")
                exit(1)
            with open(sublist) as f:
                counter = 0
                for line in f:
                    # ignore comment line
                    if not line or line[0] == '#':
                        continue
                    # print(line)
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
                    elif self.normalize_text:
                        gtText = self.normalize(lineSplit[1])
                    else:
                        gtText = lineSplit[1]
                    ignoreLine = False
                    if not include_unsupported_chars and self.injected_charlist and not self.replace_final_layer:
                        for char in gtText:
                            if char not in self.injected_charlist:
                                # print(self.injected_charlist)
                                print('a ignoring line: ' + gtText)
                                ignoreLine = True
                                break
                    if ignoreLine or len(gtText) == 0:
                        print(line)
                        continue
                    counter = counter + 1
                    for i in range(0, self.multiply):
                        partition[partition_name].append(fileName)
                        labels[partition_name].append(gtText)
                        files.append([fileName, gtText])
                    if not self.injected_charlist or self.replace_final_layer:
                        chars = chars.union(set(char for label in gtText for char in label))

                    # if counter > 100:
                    #     break
                print('found ' + str(counter) + ' lines suitable for ' + partition_name)
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
        # print(self.partition)
        return self.partition[partition][item_id]
