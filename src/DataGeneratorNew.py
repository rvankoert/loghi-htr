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
from tensorflow.keras.utils import img_to_array
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
# import elasticdeform.tf as etf
import elasticdeform
import gc

class DataGeneratorNew(tf.keras.utils.Sequence):
    DTYPE = tf.float32

    # @tf.function
    @staticmethod
    def elastic_transform(original):
        displacement = np.random.randn(2, 3, 3) * 5
        # X_deformed = elasticdeform.deform_random_grid(original)
        X_deformed = elasticdeform.deform_grid(original, displacement, axis=(0, 1), cval=0)
        return X_deformed

    # def encode_single_sample_augmented(self, img_path, label):
    #     return self.encode_single_sample(self, img_path, label, self.augment, self.do_elastic_transform, self.distort_jpeg,
    #                                      self.height, self.width, self.channels, self.do_binarize_otsu,
    #                                      self.do_binarize_sauvola, self.random_crop, self.random_width
    #                                      )

    # def encode_single_sample_clean(self, img_path, label):
    #     return self.encode_single_sample(self, img_path, label, False, False, False,
    #                                      self.height, self.width, self.channels, self.do_binarize_otsu,
    #                                      self.do_binarize_sauvola, self.random_crop, self.random_width
    #                                      )

    @staticmethod
    def sauvola(image):

        window_size = 51
        thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.1)
        binary_sauvola = np.invert(image > thresh_sauvola)*1
        # binary_sauvola = (image > thresh_sauvola)*1

        return tf.convert_to_tensor(binary_sauvola)

    # https://colab.research.google.com/drive/1CdVfa2NlkQBga1E9dBwHved36Tk7Bg61#scrollTo=Jw-NU1wbHnWA
    @staticmethod
    def otsu_thresholding(image):
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

    @staticmethod
    def check_valid_file(img, img_path):
        if img is None:
            print('can not read/find the file on disk: ' + img_path)
            print('you can use --check_missing_files to check/skip during startup for missing files')
            exit()

    @staticmethod
    def load_image(image_path, channels):
        if channels == 1:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            DataGeneratorNew.check_valid_file(img, image_path)
            img = np.expand_dims(img, -1)
            # gtImageEncoded = tf.image.encode_png(img)
            # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
        elif channels == 3:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            DataGeneratorNew.check_valid_file(img, image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            DataGeneratorNew.check_valid_file(img, image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    @staticmethod
    def augment(img, datagenerator, label, augment, elastic_transform, distort_jpeg, height, width, channels,
                             do_binarize_otsu, do_binarize_sauvola, random_crop, random_width):
        if elastic_transform:
            alpha_range = random.uniform(0, 750)
            sigma = random.uniform(0, 30)
            img = DataGeneratorNew.elastic_transform(img)
        # img = elasticdeform.deform_random_grid(img, sigma=1, points=3)
        # print (img)

        if distort_jpeg:
            if channels == 4:
                # crappy workaround for bug in shear_x where alpha causes errors
                channel1, channel2, channel3, alpha = tf.split(img, 4, axis=2)
                img = tf.concat([channel1, channel2, channel3], axis=2)
                img = tf.image.random_jpeg_quality(img, 50, 100)
                channel1, channel2, channel3 = tf.split(img, 3, axis=2)
                img = tf.concat([channel1, channel2, channel3, alpha], axis=2)
            else:
                img = tf.image.random_jpeg_quality(img, 50, 100)

        # print(img)

        # img = tf.convert_to_tensor(img)

        if do_binarize_otsu:
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # img = tf.image.rgb_to_grayscale(img)
            img = img * 255
            img = DataGeneratorNew.otsu_thresholding(img)

        if do_binarize_sauvola:
            if img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # if img.shape[2] > 1:
            #     img = tf.image.rgb_to_grayscale(img)
            # print (img.shape)
            img = DataGeneratorNew.sauvola(img)
            img = np.array(img, dtype=np.float32)
        # print(img)
        # img *= 255
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testb.png", gtImageEncoded)
        image_height = img.shape[0]
        image_width = img.shape[1]
        # img = tf.image.resize_with_pad(img, tf.shape(img)[0], tf.shape(img)[0]+tf.shape(img)[1])

        # augment=False

        if augment and channels == 3:
            randomShear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]
            img = tfa.image.shear_x(img, randomShear, replace=1.0)
        if augment and channels == 4:
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
            # random_contrast = tf.random.uniform(shape=[1], minval=0.7, maxval=1.3)[0]
            # img = tf.image.adjust_contrast(img, contrast_factor=random_contrast)

        if random_crop:
            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.8, maxval=1.0)[0]
            original_width = tf.shape(img)[1]
            original_height = tf.cast(tf.shape(img)[0], tf.float32)

            # print(random_crop)
            # print(original_height)
            crop_height = tf.cast(random_crop * original_height, tf.int32)
            crop_size = (crop_height, original_width, channels)
            img = tf.image.stateless_random_crop(img, crop_size, randomseed)

        if random_width:
            random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            image_width = int(random_width)
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            img = tf.image.resize(img, [image_height, image_width])

        # print('height1 '+ str(height) + " " + str(width))
        # img = tf.image.resize(img, [height, width], preserve_aspect_ratio=True)
        # print(label)
        label = datagenerator.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        label_width = label.shape[0]
        image_width = int((image_width/image_height)*height)
        # img = tf.image.resize(img, [height, image_width], preserve_aspect_ratio=True)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        img = tf.image.resize(img, [height, image_width])

        if self.augment and image_width < label_width*16:
            image_width = label_width * 16
            # print('setting label width '+ str(height) + " " + str(image_width))
            img = tf.image.resize_with_pad(img, height, image_width)
        return img

    @staticmethod
    def encode_single_sample(datagenerator, img_path, label, augment, elastic_transform, distort_jpeg, height, width, channels,
                             do_binarize_otsu, do_binarize_sauvola, random_crop, random_width):
        MAX_ROT_ANGLE = 10.0
        # img = tf.io.read_file(img_path)
        # img = tf.io.decode_png(img, channels=self.channels)
        # img = tf.image.convert_image_dtype(img, self.DTYPE)
        img = DataGeneratorNew.load_image(img_path, channels)

        # gtImageEncoded = tf.image.encode_png(img)
        # img = tf.image.decode_image(gtImageEncoded)
        # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
        # img *= 255
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
        # gtImageEncoded = tf.image.encode_png(img)
        # tf.io.write_file("/tmp/testb.png", gtImageEncoded)
        # exit()
        img = augment(img, datagenerator, label,augment, elastic_transform, distort_jpeg, height, channels,
                             do_binarize_otsu, do_binarize_sauvola, random_crop, random_width)



        # pad 25 pixels left and right
        # img = tf.ensure_shape(img, [self.height, None, self.channels])
        # print('height2 ' + str(height) + " " + str(image_width+50) + " " + str(label_width))
        img = tf.image.resize_with_pad(img, height, image_width+50)
        # if image_width > 6000:
        #     img = tf.image.resize_with_pad(img, height, 6000)
        # if self.channels == 1:
        #     img = 0.5 - img/255.0
        # else:
        img = 0.5 - img

        img = tf.transpose(img, perm=[1, 0, 2])
        # return {"image": img, "label": label}
        return img, label

    def __init__(self, list_IDs, labels, batch_size=1, channels=4, shuffle=True, height=32,
                 width=99999, charList=None, do_binarize_otsu=False, do_binarize_sauvola=False, augment=False,
                 elastic_transform=False, num_oov_indices=0, random_crop=False, random_width=False, distort_jpeg=False):
        """Initialization"""
        if charList is None:
            charList = []
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.channels = channels
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.charList = charList
        self.set_charlist(self.charList, num_oov_indices=num_oov_indices)
        self.do_binarize_otsu = do_binarize_otsu
        self.do_binarize_sauvola = do_binarize_sauvola
        self.augment = augment
        self.do_elastic_transform = elastic_transform
        self.random_crop = random_crop
        self.random_width = random_width
        self.distort_jpeg = distort_jpeg
        self.on_epoch_end()

    def __len__(self):
        self.on_epoch_end()
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        gc.collect()
        # tf.keras.backend.clear_session()


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

    @staticmethod
    def dynamic_padding(inp, min_size, channels):
        pad_size = min_size - inp.shape[0]
        # print('pad_size')
        # print(pad_size)
        # print(inp.shape[0])
        paddings = [[0, pad_size], [0, 0], [0, 0]]
        return tf.pad(inp, paddings, "CONSTANT", constant_values=-10)

    @staticmethod
    def dynamic_padding2(inp, min_size):

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
        max_size_x = 0
        max_size_y = 0
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
                item = DataGeneratorNew.encode_single_sample(self, filename, label, self.augment, self.do_elastic_transform, self.distort_jpeg,
                                         self.height, self.width, self.channels, self.do_binarize_otsu,
                                         self.do_binarize_sauvola, self.random_crop, self.random_width)
            else:
                # item = self.encode_single_sample_clean(filename, label)
                item = DataGeneratorNew.encode_single_sample(self, filename, label, False, False, False,
                                         self.height, self.width, self.channels, self.do_binarize_otsu,
                                         self.do_binarize_sauvola, self.random_crop, self.random_width
                                         )
            # item = self.get_baselines(ID)
            X.append(item[0])
            Y.append(item[1])
            size_x = item[0].shape[0]
            size_y = len(item[1])
            if size_x > max_size_x:
                max_size_x = size_x
            if size_y > max_size_y:
                max_size_y = size_y
        for i, ID in enumerate(list_IDs_temp):
            # gtImageEncoded = tf.image.encode_png(tf.transpose(tf.cast(-X[i], dtype="uint8"), perm=[1, 0, 2]))
            # tf.io.write_file("/tmp/testa.png", gtImageEncoded)
            X[i] = self.dynamic_padding(X[i], max_size_x, self.channels)
            # gtImageEncoded = tf.image.encode_png(tf.transpose(tf.cast(-X[i], dtype="uint8"), perm=[1, 0, 2]))
            # tf.io.write_file("/tmp/testb.png", gtImageEncoded)
            # exit()

            # print(Y[i])
            Y[i] = self.dynamic_padding2(Y[i], max_size_y)
            # print(Y[i])
            # exit()
        X = tf.convert_to_tensor(X)
        Y = tf.convert_to_tensor(Y)
        return X, Y

    def set_charlist(self, chars, use_mask=False, num_oov_indices=0):
        self.charList = chars
        if num_oov_indices > 0:
            self.charList.insert(1, '[UNK]')
        if not self.charList:
            return
        if use_mask:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token='', oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token='', encoding="UTF-8",
                invert=True
            )
        else:
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.charList), num_oov_indices=num_oov_indices, mask_token=None, oov_token='[UNK]', encoding="UTF-8"
            )
            # Mapping integers back to original characters
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None, encoding="UTF-8",
                invert=True
            )

