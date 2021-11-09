from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras import layers
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    DTYPE = np.float32

    def encode_single_sample_augmented(self, img_path, label):
        return self.encode_single_sample(img_path, label, True)

    def encode_single_sample_clean(self, img_path, label):
        return self.encode_single_sample(img_path, label, False)

    def encode_single_sample(self, img_path, label, augment):
        MAX_ROT_ANGLE = 10.0
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=self.channels)
        img = tf.image.convert_image_dtype(img, self.DTYPE)

        print("img.shape[0]")
        print(tf.shape(img)[0])
        img = tf.image.resize(img, [self.height, self.width], preserve_aspect_ratio=True)
        img = 1.0 - img
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # img = tf.image.resize_with_pad(img, 51, 1024)
        imageWidth = tf.shape(img)[1]
        labelWidth = tf.shape(label)[0]
        if imageWidth < labelWidth*4:
            img = tf.image.resize_with_pad(img, self.height, labelWidth*4)

        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img, "label": label}

    def __init__(self, list_IDs, labels, batch_size=1, dim=(751, 51, 4), channels=4, shuffle=True, height=32, width=99999, charList=[]):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.channels = channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.height = height
        self.width = width
        self.on_epoch_end()
        self.charList = charList
        self.set_charlist(self.charList)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_IDs,self.labels))

    def getGenerator(self):

        train_dataset = self.dataset
        if self.shuffle:
             train_dataset = train_dataset.shuffle(len(self.dataset))
        train_dataset = (
            train_dataset
            .map(
                self.encode_single_sample_augmented, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .padded_batch(self.batch_size, padded_shapes={
                'image': [None, None, None],
                'label': [None]
            })
            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return train_dataset


    def set_charlist(self, chars):
        self.charList = chars
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.charList), num_oov_indices=0, mask_token=None, oov_token='[UNK]'
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None, invert=True
        )
