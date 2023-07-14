# import the necessary packages
import math

# from helpers import benchmark
from tensorflow.data import AUTOTUNE
import tensorflow as tf
import argparse
import time
import random
from utils import Utils
import elasticdeform.tf as etf
import tensorflow_addons as tfa

class DataGeneratorNew2(tf.keras.utils.Sequence):

    def __init__(self,
                 utils,
                 batchSize,
                 height=64,
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 augment=False,
                 do_elastic_transform=False,
                 random_crop=False,
                 random_width=False,
                 distort_jpeg=False,
                 channels=1,
                 do_random_shear=False
                 ):
        print(height)

        self.batchSize = batchSize
        self.do_binarize_sauvola = do_binarize_sauvola
        self.do_binarize_otsu = do_binarize_otsu
        self.do_elastic_transform = do_elastic_transform
        self.random_crop = random_crop
        self.random_width = random_width
        self.distort_jpeg = distort_jpeg
        self.utils = utils
        self.height = height
        self.channels = channels
        self.do_random_shear = do_random_shear

    # @staticmethod
    # def elastic_transform(original, seed=42):
    #     # displacement = tf.random.uniform(shape=[3], minval=0, maxval=3)[0] * 5
    #     #
    #     np.random.seed(seed)
    #     displacement = np.random.randn(2, 3, 3) * 5
    #     # X_deformed = elasticdeform.deform_random_grid(original)
    #     X_deformed = elasticdeform.deform_grid(original, displacement, axis=(0, 1), cval=0)
    #     return X_deformed

    def elastic_transform(self, original):

        displacement_val = tf.random.normal([2, 3, 3]) * 5
        # X_val = numpy.random.rand(200, 300)
        # dY_val = numpy.random.rand(200, 300)

        # construct TensorFlow input and top gradient
        # displacement = tf.Variable(displacement_val)
        # X = tf.Variable(original)
        # dY = tf.Variable(dY_val)

        # the deform_grid function is similar to the plain Python equivalent,
        # but it accepts and returns TensorFlow Tensors
        X_deformed = etf.deform_grid(original, displacement_val, axis=(0, 1), order=3)

        # # the gradient w.r.t. X can be computed in the normal TensorFlow manner
        # [dX] = tf.gradients(X_deformed, X, dY)
        return X_deformed

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

        image_width = tf.shape(image)[1]
        image_height = tf.shape(image)[0]
        if self.do_elastic_transform:
            # print(image)
            image = self.elastic_transform(image)
            print(image)

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

        print(image)
        if self.random_width:
            random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]
            random_width *= float(image_width)
            image_width = int(random_width)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, [image_height, image_width])

        image = tf.image.resize_with_pad(image, self.height, image_width+50)

        if self.do_elastic_transform:
            image = tf.image.resize_with_pad(image, self.height, image_width + 64 + 50)
            random_shear = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

            if self.channels == 4:
                # crappy workaround for bug in shear_x where alpha causes errors
                channel1, channel2, channel3, alpha = tf.split(image, 4, axis=2)
                image = tf.concat([channel1, channel2, channel3], axis=2)
                image = tfa.image.shear_x(image, random_shear, replace=0)
                image2 = tf.concat([alpha, alpha, alpha], axis=2)
                image2 = tfa.image.shear_x(image2, random_shear, replace=0)
                channel1, channel2, channel3 = tf.split(image, 3, axis=2)
                alpha, alpha, alpha = tf.split(image2, 3, axis=2)
                image = tf.concat([channel1, channel2, channel3, alpha], axis=2)
            elif self.channels == 3:
                image = tfa.image.shear_x(image, random_shear, replace=0)
            else:
                # channel1 = tf.split(image, 1, axis=-1)
                image = tf.concat([image, image, image], axis=2)
                image = tfa.image.shear_x(image, random_shear, replace=0)
                image, image, image = tf.split(image, 3, axis=2)

        label = imagePath[1]
        encodedLabel = self.utils.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        label_counter = 0
        lastchar =None
        for char in encodedLabel:
            label_counter += 1
            if char == lastchar:
                label_counter += 1
            lastchar = char
        label_width = label_counter
        if image_width < label_width*16:
            image_width = label_width * 16
            # print('setting label width '+ str(height) + " " + str(image_width))
            image = tf.image.resize_with_pad(image, self.height, image_width)

        # time.sleep(2.0)
        # gtImageEncoded = tf.image.encode_png(tf.cast(image*255, dtype="uint8"))
        # tf.io.write_file("/tmp/testa.png", gtImageEncoded)

        image = 0.5 - image
        image = tf.transpose(image, perm=[1, 0, 2])

        # print('label')
        # print(label)
        # encodedLabel = self.utils.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return image, encodedLabel

