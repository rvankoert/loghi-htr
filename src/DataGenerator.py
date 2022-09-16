from __future__ import division
from __future__ import print_function

import math

import cv2
import keras.backend
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from keras.preprocessing.image import img_to_array
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


class DataGenerator(tf.keras.utils.Sequence):
    DTYPE = tf.float32

    # @staticmethod
    # @tf.function
    # def elastic_transform(self, original, alpha_range, sigma, random_state=None):
    #
    #     displacement_val = np.random.randn(2, 3, 3) * 5
    #     # X_val = np.array(original)
    #     # print(X_val)
    #
    #     # construct TensorFlow input and top gradient
    #     displacement = tf.Variable(displacement_val)
    #     X = tf.Variable(original)
    #
    #     # the deform_grid function is similar to the plain Python equivalent,
    #     # but it accepts and returns TensorFlow Tensors
    #     X_deformed = etf.deform_grid(X, displacement, order=3)
    #     print(tf.shape(X_deformed)[0])
    #     return X_deformed

    def encode_single_sample_augmented(self, img_path, label):
        return self.encode_single_sample(img_path, label, self.dataAugmentation)

    def encode_single_sample_clean(self, img_path, label):
        return self.encode_single_sample(img_path, label, False)

    @staticmethod
    def sauvola(image):

        window_size = 51
        thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.1)
        binary_sauvola = np.invert(image > thresh_sauvola)*1

        return tf.convert_to_tensor(binary_sauvola)

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    @staticmethod
    def otsu_thresholding(image):
        image = tf.convert_to_tensor(image, name="image")
        # image = tf.squeeze(image)
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")
        # print (image.shape)

        if image.dtype != tf.int32:
            image = tf.cast(image, tf.int32)

        r, c, detected_channels = image.shape
        hist = tf.math.bincount(image, dtype=tf.int32)

        if len(hist) < 256:
            hist = tf.concat([hist, [0] * (256 - len(hist))], 0)

        current_max, threshold = 0, 0
        total = r * c

        spre = [0] * 256
        sw = [0] * 256
        spre[0] = int(hist[0])

        for i in range(1, 256):
            spre[i] = spre[i - 1] + int(hist[i])
            sw[i] = sw[i - 1] + (i * int(hist[i]))

        for i in range(256):
            if total - spre[i] == 0:
                break

            meanB = 0 if int(spre[i]) == 0 else sw[i] / spre[i]
            meanF = (sw[255] - sw[i]) / (total - spre[i])
            varBetween = (total - spre[i]) * spre[i] * ((meanB - meanF) ** 2)

            if varBetween > current_max:
                current_max = varBetween
                threshold = i

        final = tf.where(image > threshold, 0, 1)
        # final = tf.expand_dims(final, -1)
        return final

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    @staticmethod
    def adaptive_thresholding(image):
        image = tf.convert_to_tensor(image, name="image")
        window = 40
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")

        if not isinstance(window, int):
            raise ValueError("Window size value must be an integer.")
        # print(image.shape)
        r, c, channels = image.shape
        if window > min(r, c):
            raise ValueError("Window size should be lesser than the size of the image.")

        if rank == 3:
            image = tf.image.rgb_to_grayscale(image)
            image = tf.squeeze(image, 2)

        image = tf.image.convert_image_dtype(image, tf.dtypes.float32)

        i = 0
        final = tf.zeros((r, c))
        while i < r:
            j = 0
            r1 = min(i + window, r)
            while j < c:
                c1 = min(j + window, c)
                cur = image[i:r1, j:c1]
                thresh = tf.reduce_mean(cur)
                new = tf.where(cur > thresh, 255.0, 0.0)

                s1 = [x for x in range(i, r1)]
                s2 = [x for x in range(j, c1)]
                X, Y = tf.meshgrid(s2, s1)
                ind = tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])], axis=1)

                final = tf.tensor_scatter_nd_update(final, ind, tf.reshape(new, [-1]))
                j += window
            i += window
        final = tf.expand_dims(final, -1)
        return final

    def encode_single_sample(self, img_path, label, augment):
        MAX_ROT_ANGLE = 10.0
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=self.channels)
        img = tf.image.convert_image_dtype(img, self.DTYPE)
        # img = cv2.imread('/scratch/train_data_ocr/linestripsclean/08/49/0849aafc-c320-4053-85ad-39b67f6e3e22.png')
        # img = elasticdeform.deform_random_grid(img, sigma=25, points=3)
        # if augment:
        #     alpha_range = random.uniform(0, 750)
        #     sigma = random.uniform(0, 30)
        #     img = self.elastic_transform(self, img, alpha_range, sigma)
        # print(img)

        # img = tf.convert_to_tensor(img)

        if self.do_binarize_otsu:
            if img.shape[2] > 1:
                img = tf.image.rgb_to_grayscale(img)
            img = img * 255
            img = self.otsu_thresholding(img)

        if self.do_binarize_sauvola:
            if img.shape[2] > 1:
                img = tf.image.rgb_to_grayscale(img)
            # sess = keras.backend.get_session()
            # with sess.as_default():
            img = self.sauvola(img)

        img = tf.image.resize_with_pad(img, tf.shape(img)[0], tf.shape(img)[0]+tf.shape(img)[1])

        # augment=False

        if augment and self.channels < 4:
            randomShear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]
            img = tfa.image.shear_x(img, randomShear, replace=1.0)
        if augment and self.channels == 4:
            # crappy workaround for bug in shear_x where alpha causes errors
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

            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.8, maxval=1.0)[0]
            original_height = tf.cast(tf.shape(img)[0], tf.float32)
            original_width = float(tf.shape(img)[1])
            # print(random_crop)
            # print(original_height)
            crop_height = random_crop * original_height
            crop_size = (crop_height, original_width, img.shape[2])
            img = tf.image.stateless_random_crop(img, crop_size, randomseed)

            image_width = tf.shape(img)[1]
            image_height = tf.shape(img)[0]
            random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            random_width = int(random_width)
            img = tf.image.resize(img, [image_height, random_width])

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
                 width=99999, charList=None, do_binarize_otsu=False, do_binarize_sauvola=False, data_augmentation=True,
                 num_oov_indices=0):
        'Initialization'
        if charList is None:
            charList = []
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
        self.set_charlist(self.charList, num_oov_indices=num_oov_indices)
        self.do_binarize_otsu = do_binarize_otsu
        self.do_binarize_sauvola = do_binarize_sauvola
        self.dataset = tf.data.Dataset.from_tensor_slices((self.list_IDs, self.labels))
        self.dataAugmentation = data_augmentation

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
                    # .map(lambda x, y: tf.py_function(self.encode_single_sample_augmented, [x,y],
                    # [tf.float32,tf.int64]))
                    # .map(lambda x:
                    #     tf.py_function(func=self.encode_single_sample_augmented,
                    #                    inp=[x],
                    #                    Tout=tf.string)
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

    def set_charlist(self, chars, use_mask=False, num_oov_indices=0):
        self.charList = chars
        # self.charList.append('[UNK]')
        if not self.charList:
            return
        if use_mask:
            self.char_to_num = layers.experimental.preprocessing.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token='', oov_token='[UNK]'
            )
            # Mapping integers back to original characters
            self.num_to_char = layers.experimental.preprocessing.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token='',
                invert=True
            )
        else:
            self.char_to_num = layers.experimental.preprocessing.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token=None, oov_token='[UNK]'
            )
            # Mapping integers back to original characters
            self.num_to_char = layers.experimental.preprocessing.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None,
                invert=True
            )

