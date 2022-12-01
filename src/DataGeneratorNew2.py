# import the necessary packages
import math

# from helpers import benchmark
from tensorflow.data import AUTOTUNE
import tensorflow as tf
import argparse
import time
import random
from utils import Utils


class DataGeneratorNew2(tf.keras.utils.Sequence):

    def __init__(self, utils, batchSize, height=64,
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 normalize_text=False,
                 multiply=1,
                 augment=True,
                 elastic_transform=False,
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
                 channels=1
                 ):
        self.batchSize = batchSize
        self.do_binarize_sauvola = do_binarize_sauvola
        self.do_binarize_otsu = do_binarize_otsu
        self.normalize_text = normalize_text
        self.multiply = multiply
        self.dataAugmentation = augment
        self.elastic_transform = elastic_transform
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
        self.utils = utils
        self.height = height
        self.channels = channels

    # @staticmethod
    # def elastic_transform(original, seed=42):
    #     # displacement = tf.random.uniform(shape=[3], minval=0, maxval=3)[0] * 5
    #     #
    #     np.random.seed(seed)
    #     displacement = np.random.randn(2, 3, 3) * 5
    #     # X_deformed = elasticdeform.deform_random_grid(original)
    #     X_deformed = elasticdeform.deform_grid(original, displacement, axis=(0, 1), cval=0)
    #     return X_deformed

    def load_images(self, imagePath):
        image = tf.io.read_file(imagePath[0])
        image = tf.image.decode_png(image, channels=self.channels)
        image = tf.image.resize(image, (self.height, 99999), preserve_aspect_ratio=True) / 255.0
        if self.distort_jpeg:
            if self.channels == 4:
                # crappy workaround for bug in shear_x where alpha causes errors
                channel1, channel2, channel3, alpha = tf.split(image, 4, axis=2)
                image = tf.concat([channel1, channel2, channel3], axis=2)
                image = tf.image.random_jpeg_quality(image, 50, 100)
                channel1, channel2, channel3 = tf.split(image, 3, axis=2)
                image = tf.concat([channel1, channel2, channel3, alpha], axis=2)
            else:
                image = tf.image.random_jpeg_quality(image, 20, 100)

        if self.random_crop:
            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.6, maxval=1.0)[0]
            original_width = tf.shape(image)[1]
            original_height = tf.cast(tf.shape(image)[0], tf.float32)

            # print(random_crop)
            # print(original_height)
            crop_height = tf.cast(random_crop * original_height, tf.int32)
            crop_size = (crop_height, original_width, self.channels)
            image = tf.image.stateless_random_crop(image, crop_size, randomseed)

        image_width = tf.shape(image)[1]
        image_height = tf.shape(image)[0]

        if self.random_width:
            random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            image_width = int(random_width)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, [image_height, image_width])

        image = tf.image.resize_with_pad(image, self.height, image_width+50)
        image = 0.5 - image
        image = tf.transpose(image, perm=[1, 0, 2])

        label = imagePath[1]
        # print('label')
        # print(label)
        encodedLabel = self.utils.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return image, encodedLabel

