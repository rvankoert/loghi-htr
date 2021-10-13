from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa


class DataLoader:
    DTYPE = 'float32'

    dataAugmentation = False
    currIdx = 0
    charList = []
    samples = []
    validation_dataset = [];

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        # f = open('/scratch/train_data_htr/linestripsnew/all.txt')
        # f = open('/home/rutger/training_all2.txt')
        f = open('/home/rutger/training_all_ijsberg.txt')

        chars = set()
        bad_samples = []
        i=0
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue
            # print(line)
            lineSplit = line.strip().split('\t')
            assert len(lineSplit) >= 1

            fileName = lineSplit[0]

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[1:]), maxTextLen).replace("|", " ")
            if not gtText:
                continue
            gtText = gtText.strip()
            if not gtText:
                continue
            # chars = chars.union(set(list(gtText)))
            chars = chars.union(set(char for label in gtText for char in label))
            # check if image is not empty
            if not os.path.exists(fileName):
                # print(fileName)
                continue

            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                print("bad sample: "+ lineSplit[0])
                continue
            img = cv2.imread(fileName)
            height, width, channels = img.shape
            # print (width *(height/ 32))
            if height < 32 or width < 32 or width /(height / 32) < len(gtText):
                print(fileName)
                # os.remove(fileName)
                continue

            # put sample into list
            self.samples.append((gtText, fileName))
            i = i+1
            if i%1000==0:
                print(i)
        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if len(bad_samples) > 0:
            print("Warning, damaged images found:", bad_samples)
        print("load textlines")

        # split into training and validation set: 95% - 5%
        random.seed(42)
        # random.shuffle(self.samples)

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 64

        # list of all chars in dataset
        self.charList = sorted(list(chars))

        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.charList), num_oov_indices=0, mask_token=None, oov_token='[UNK]'
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None, invert=True
        )

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def encode_single_sample_augmented(self, img_path, label):
        return self.encode_single_sample(img_path, label, True)

    def encode_single_sample_clean(self, img_path, label):
        return self.encode_single_sample(img_path, label, False)

    def encode_single_sample(self, img_path, label, augment):
        MAX_SHEAR_LEVEL = 0.1
        MAX_ROT_ANGLE = 10.0
        HSHIFT, VSHIFT = 5., 5.  # max. number of pixels to shift(translation) horizontally and vertically
        MAX_ROT_ANGLE = np.pi * MAX_ROT_ANGLE / 180  # in radians

        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, DataLoader.DTYPE)
        # 4. Resize to the desired size
        # height =32/img.shape
        print("img.shape[1]")
        print(img)
        # img = tf.image.resize_with_pad(img, 32, 1024)
        img = tf.image.resize(img, [32, 2048], preserve_aspect_ratio=True)
        # if augment:
        #     # img = tfa.image.rotate(img, MAX_ROT_ANGLE * tf.random.uniform([], dtype=DataLoader.DTYPE))  # rotation
        #     img = tfa.image.translate(img, [HSHIFT * tf.random.uniform(shape=[], minval=-1, maxval=1),
        #                                     VSHIFT * tf.random.uniform(shape=[], minval=-1,
        #                                                                maxval=1)])  # [dx dy] shift/translation
        #     img = tfa.image.transform(img,
        #                               [1.0, MAX_SHEAR_LEVEL * tf.random.uniform(shape=[], minval=-1, maxval=1), 0.0,
        #                                MAX_SHEAR_LEVEL * tf.random.uniform(shape=[], minval=-1, maxval=1), 1.0, 0.0,
        #                                0.0,
        #                                0.0])
        #     # img = tf.image.random_hue(img, 0.08)
        #     # img = tf.image.random_saturation(img, 0.6, 1.6)
        #     img = tf.image.random_brightness(img, 0.05)
        #     img = tf.image.random_contrast(img, 0.7, 1.3)

        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    def split_data(self, images, labels, train_size=0.99, shuffle=True):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid

    def getValidationDataSet(self):
        return self.validation_dataset

    def getTrainDataSet(self):
        "iterator"

        #		label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        # gtTexts = [char_to_num(tf.strings.unicode_split(self.samples[i].gtText, input_encoding="UTF-8")) for i in batchRange]
        #		gtTexts = [char_to_num(tf.strings.unicode_split(self.samples[i].gtText, input_encoding="UTF-8")) for i in batchRange]
        # gtTexts = [char_to_num(tf.strings.unicode_split("test", input_encoding="UTF-8")) for i in batchRange]
        gtTexts = [self.samples[i][0] for i in range(len(self.samples))]
        # gtTexts = ["test test " for i in range(len(self.samples))]

        max_length = max([len(label) for label in gtTexts])
        print("max_length: " + str(max_length))
        # gtTexts = [(str(i)+"a123456").ljust(10,'\0') for i in batchRange]

        #		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_UNCHANGED), self.imgSize, self.dataAugmentation) for i in batchRange]
        imgs = [self.samples[i][1] for i in range(len(self.samples))]

        x_train, x_valid, y_train, y_valid = self.split_data(np.array(imgs), np.array(gtTexts))
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        train_dataset = (
            train_dataset.map(
                self.encode_single_sample_augmented, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(self.batchSize)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        print(validation_dataset)
        self.validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample_clean, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(self.batchSize)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return train_dataset
