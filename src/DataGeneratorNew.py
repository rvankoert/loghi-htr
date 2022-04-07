from __future__ import division
from __future__ import print_function

import math

import cv2
import elasticdeform
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
# import elasticdeform.tf as etf
import numpy, imageio, elasticdeform

class DataGeneratorNew(tf.keras.utils.Sequence):
    DTYPE = tf.float32

    # @tf.function
    def elastic_transform(self, original):
        displacement = np.random.randn(2, 3, 3) * 5
        # X_deformed = elasticdeform.deform_random_grid(original)
        X_deformed = elasticdeform.deform_grid(original, displacement, axis=(0, 1), cval=0)
        return X_deformed

    def encode_single_sample_augmented(self, img_path, label):
        return self.encode_single_sample(img_path, label, self.augment, self.do_elastic_transform)

    def encode_single_sample_clean(self, img_path, label):
        return self.encode_single_sample(img_path, label, False, False)

    def sauvola(self, image):

        window_size = 51
        thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.1)
        binary_sauvola = np.invert(image > thresh_sauvola)*1
        # binary_sauvola = (image > thresh_sauvola)*1

        return tf.convert_to_tensor(binary_sauvola)

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    def otsu_thresholding(self, image):
        image = tf.convert_to_tensor(image, name="image")
        # image = tf.squeeze(image)
        rank = image.shape.rank
        if rank != 2 and rank != 3:
            raise ValueError("Image should be either 2 or 3-dimensional.")

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
    def adaptive_thresholding(self, image):
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

    def encode_single_sample(self, img_path, label, augment, elastic_transform):
        MAX_ROT_ANGLE = 10.0
        # img = tf.io.read_file(img_path)
        # img = tf.io.decode_png(img, channels=self.channels)
        # img = tf.image.convert_image_dtype(img, self.DTYPE)
        if self.channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, -1)
        elif self.channels == 3:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # if not img:
            #     print (img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
        # img *= 255
        if elastic_transform:
            alpha_range = random.uniform(0, 750)
            sigma = random.uniform(0, 30)
            img = self.elastic_transform(img)
        # img = elasticdeform.deform_random_grid(img, sigma=1, points=3)
        # print (img)
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testb.png", gtImageEncoded)

        # print(img)

        # img = tf.convert_to_tensor(img)

        if self.do_binarize_otsu:
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img = tf.image.rgb_to_grayscale(img)
            img = img * 255
            img = self.otsu_thresholding(img)

        if self.do_binarize_sauvola:
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # if img.shape[2] > 1:
            #     img = tf.image.rgb_to_grayscale(img)
            # print (img.shape)
            img = self.sauvola(img)
            img = np.array(img, dtype=np.float32)
        # print(img)
        # img *= 255
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testb.png", gtImageEncoded)

        # img = tf.image.resize_with_pad(img, tf.shape(img)[0], tf.shape(img)[0]+tf.shape(img)[1])

        # augment=False

        if augment and self.channels == 3:
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

        if self.random_crop:
            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.8, maxval=1.0)[0]
            original_height = tf.cast(tf.shape(img)[0], tf.float32)
            original_width = float(tf.shape(img)[1])
            # print(random_crop)
            # print(original_height)
            crop_height = random_crop * original_height
            crop_size = (crop_height, original_width, img.shape[2])
            img = tf.image.stateless_random_crop(img, crop_size, randomseed)

        if self.random_width:
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
        if image_width < label_width*16:
            img = tf.image.resize_with_pad(img, self.height, label_width*16)

        image_width = tf.shape(img)[1]
        # pad 25 pixels left and right
        img = tf.image.resize_with_pad(img, self.height, image_width+50)
        if image_width > 6000:
            img = tf.image.resize_with_pad(img, self.height, 6000)
        img = 0.5 - img

        img = tf.transpose(img, perm=[1, 0, 2])
        # return {"image": img, "label": label}
        return img, label

    def __init__(self, list_IDs, labels, batch_size=1, dim=(751, 51, 4), channels=4, shuffle=True, height=32,
                 width=99999, charList=[], do_binarize_otsu=False, do_binarize_sauvola=False, augment=False,
                 elastic_transform=False, num_oov_indices=0, random_crop=False, random_width=False):
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
        self.set_charlist(self.charList, num_oov_indices=num_oov_indices)
        self.do_binarize_otsu = do_binarize_otsu
        self.do_binarize_sauvola = do_binarize_sauvola
        self.augment = augment
        self.do_elastic_transform = elastic_transform
        self.random_crop = random_crop
        self.random_width = random_width

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_file(self, index):
        # Find list of IDs
        return self.list_IDs[index]

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(indexes)
        return X, Y


    def dynamic_padding(self, inp, min_size):

        pad_size = min_size - inp.shape[0]
        # print('pad_size')
        # print(pad_size)
        # print(inp.shape[0])
        paddings = [[0, pad_size], [0, 0], [0, 0]]
        return tf.pad(inp, paddings, "CONSTANT", constant_values=-10)

    def dynamic_padding2(self, inp, min_size):

        pad_size = min_size - inp.shape[0]
        # print('pad_size')
        # print(pad_size)
        # print(inp.shape[0])
        paddings = [[0, pad_size]]
        return tf.pad(inp, paddings, "CONSTANT")

    def __data_generation(self, list_IDs_temp):

        # print (list_IDs_temp[0])
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        Y = []

        # TODO: make this multithreading
        # Generate data
        # print(list_IDs_temp)
        max_size_x= 0
        max_size_y= 0
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            label = self.labels[ID]
            filename = self.list_IDs[ID]
            # print(self.labels)
            # print(ID)
            # print(label)
            # print(filename)
            if self.shuffle:
                # print(ID)
                item = self.encode_single_sample_augmented(filename, label)
            else:
                item = self.encode_single_sample_clean(filename, label)
            # item = self.get_baselines(ID)
            X.append(item[0])
            Y.append(item[1])
            size_x = item[0].shape[0]
            size_y = len(item[1])
            if size_x>max_size_x:
                max_size_x = size_x
            if size_y>max_size_y:
                max_size_y = size_y
        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.dynamic_padding(X[i], max_size_x)
            Y[i] = self.dynamic_padding2(Y[i], max_size_y)
        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)
        return X, Y



    def set_charlist(self, chars, use_mask = False, num_oov_indices=0):
        self.charList = chars
        if num_oov_indices>0:
            self.charList.insert(1, '[UNK]')
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

