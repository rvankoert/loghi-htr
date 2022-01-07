from __future__ import division
from __future__ import print_function

import math

import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from keras.preprocessing.image import img_to_array


class DataGenerator(tf.keras.utils.Sequence):
    DTYPE = tf.float32

    @staticmethod
    @tf.function
    def elastic_transform(original, alpha_range, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.

       # Arguments
           image: Numpy array with shape (height, width, channels).
           alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
               Controls intensity of deformation.
           sigma: Float, sigma of gaussian filter that smooths the displacement fields.
           random_state: `numpy.random.RandomState` object for generating displacement fields.
        """

        if random_state is None:
            random_state = np.random.RandomState(None)

        if np.isscalar(alpha_range):
            alpha = alpha_range
        else:
            alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

        shape = tf.shape(original)
        randomx = random_state.rand(shape)
        randomy = random_state.rand(shape)
        dx = gaussian_filter((randomx * 2 - 1), sigma) * alpha
        dy = gaussian_filter((randomy * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))
        original = map_coordinates(original, indices, order=3, mode='reflect').reshape(shape)

        # x, y, z = np.meshgrid(np.arange(gtShape[0]), np.arange(gtShape[1]), np.arange(gtShape[2]), indexing='ij')
        # indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))
        # gtImageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(gtImage,dtype=tf.uint8))
        # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
        # imageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(original,dtype=tf.uint8))
        # tf.io.write_file("/tmp/testb.png", imageEncoded)
        return original

    def encode_single_sample_augmented(self, img_path, label):
        return self.encode_single_sample(img_path, label, True)

    def encode_single_sample_clean(self, img_path, label):
        return self.encode_single_sample(img_path, label, False)

    def encode_single_sample(self, img_path, label, augment):
        MAX_ROT_ANGLE = 10.0
        alpha_range = 500
        sigma = 20
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=self.channels)
        img = tf.image.convert_image_dtype(img, self.DTYPE)
        img = tf.image.resize_with_pad(img, tf.shape(img)[0], tf.shape(img)[0]+tf.shape(img)[1])

        # augment=False

        if augment and self.channels < 4:
            randomShear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]
            img = tfa.image.shear_x(img, randomShear, replace=1.0)
        if augment and self.channels == 4:
            #crappy workaround for bug in shear_x where alpha causes errors
            channel1, channel2, channel3, alpha = tf.split(img, 4, axis=2)
            randomShear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]
            alpha = tf.concat([channel1, channel2, alpha], axis=2)  # add two dummy channels
            alpha = tfa.image.shear_x(alpha, randomShear, replace=0)
            img = tf.concat([channel1, channel2, channel3], axis=2)
            channel1, channel2, alpha = tf.split(alpha, 3, axis=2)
            img = tfa.image.shear_x(img, randomShear, replace=0)
            channel1, channel2, channel3 = tf.split(img, 3, axis=2)
            img = tf.concat([channel1, channel2, channel3, alpha], axis=2)
            # gtImageEncoded = tf.image.encode_png(tf.image.convert_image_dtype(img, dtype=tf.uint8))
            # tf.io.write_file("/tmp/testa.png", gtImageEncoded)

        if augment:
            random_brightness = tf.random.uniform(shape=[1], minval=-0.5, maxval=0.5)[0]
            img = tf.image.adjust_brightness(img, delta=random_brightness)
            random_contrast = tf.random.uniform(shape=[1], minval=0.7, maxval=1.3)[0]
            img = tf.image.adjust_contrast(img, random_contrast)
            image_width = tf.shape(img)[1]
            image_height = tf.shape(img)[0]
            random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            random_width = int(random_width)
            img = tf.image.resize(img, [image_height, random_width])

        #     img = self.elastic_transform(img, alpha_range, sigma)

        img = tf.image.resize(img, [self.height, self.width], preserve_aspect_ratio=True)
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        image_height = tf.shape(img)[0]
        image_width = tf.shape(img)[1]
        label_width = tf.shape(label)[0]
        # # #     img = tf.image.resize_with_pad(img, image_height, image_width)
        #     print("img.shape[0]")
        #     print(tf.shape(img)[0])

        # img = tf.image.resize_with_pad(img, 51, 1024)
        if image_width < label_width*32:
            img = tf.image.resize_with_pad(img, self.height, label_width*32)

        image_width = tf.shape(img)[1]
        # pad 50 pixels left and right
        img = tf.image.resize_with_pad(img, self.height, image_width+100)
        img = 0.5 - img

        img = tf.transpose(img, perm=[1, 0, 2])
        # return {"image": img, "label": label}
        return img, label

    def __init__(self, list_IDs, labels, batch_size=1, dim=(751, 51, 4), channels=4, shuffle=True, height=32,
                 width=99999, charList=[]):
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
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_IDs, self.labels))

    def getGenerator(self):

        train_dataset = self.dataset
        # if self.shuffle:
        #     train_dataset = train_dataset.shuffle(len(self.dataset))
        #     train_dataset = (
        #         train_dataset
        #         .map(
        #             self.encode_single_sample_augmented, num_parallel_calls=tf.data.experimental.AUTOTUNE
        #         )
        #         .padded_batch(self.batch_size, padded_shapes={
        #             'image': [None, None, None],
        #             'label': [None]
        #         })
        #         .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #     )
        # else:
        #     train_dataset = (
        #         train_dataset
        #         .map(
        #             self.encode_single_sample_clean, num_parallel_calls=tf.data.experimental.AUTOTUNE
        #         )
        #         .padded_batch(self.batch_size, padded_shapes={
        #             'image': [None, None, None],
        #             'label': [None]
        #         })
        #         .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        #     )
        if self.shuffle:
            train_dataset = train_dataset.shuffle(len(self.dataset))
            train_dataset = (
                train_dataset
                .map(
                    self.encode_single_sample_augmented, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .padded_batch(self.batch_size, padded_shapes=(
                    [None, None, None],
                    [None]
                ), padding_values=(-10.0, tf.cast(0, tf.int64))
                )
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )
        else:
            train_dataset = (
                train_dataset
                .map(
                    self.encode_single_sample_clean, num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
                .padded_batch(self.batch_size, padded_shapes=(
                    [None, None, None],
                    [None]
                ), padding_values=(-10.0,  tf.cast(0, tf.int64))
                )
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )
        return train_dataset


    def set_charlist(self, chars, use_mask = False):
        self.charList = chars
        if use_mask:
            self.char_to_num = layers.experimental.preprocessing.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=0, mask_token='', oov_token='[UNK]'
            )
            # Mapping integers back to original characters
            self.num_to_char = layers.experimental.preprocessing.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token='',
                invert=True
            )
        else:
            self.char_to_num = layers.experimental.preprocessing.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=0, mask_token=None, oov_token='[UNK]'
            )
            # Mapping integers back to original characters
            self.num_to_char = layers.experimental.preprocessing.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None,
                invert=True
            )

