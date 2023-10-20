# Imports

# > Standard Library
import random

# > Local dependencies

# > Third party libraries
import cv2
import tensorflow as tf
import elasticdeform.tf as etf
import tensorflow_addons as tfa
from skimage.filters import threshold_otsu, threshold_sauvola

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 utils,
                 batch_size,
                 height=64,
                 do_binarize_sauvola=False,
                 do_binarize_otsu=False,
                 do_elastic_transform=False,
                 random_crop=False,
                 random_width=False,
                 distort_jpeg=False,
                 channels=1,
                 do_random_shear=False
                 ):
        print(height)

        self.batch_size = batch_size
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

    def elastic_transform(self, original):
        displacement_val = tf.random.normal([2, 3, 3]) * 5
        X_deformed = etf.deform_grid(original, displacement_val, axis=(0, 1), order=3)
        return X_deformed

    def binarize_sauvola(self, tensor):
        np_array = tensor.numpy()
        window_size = 51

        sauvola_thresh = threshold_sauvola(np_array, window_size=window_size)
        binary_sauvola = (np_array > sauvola_thresh) * 1

        return tf.convert_to_tensor(binary_sauvola)

    def binarize_otsu(self, tensor):
        np_array = tensor.numpy()

        np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)

        otsu_threshold = threshold_otsu(np_array)

        return tf.convert_to_tensor((np_array > otsu_threshold) * 1)

    def load_images(self, image_path):
        image = tf.io.read_file(image_path[0])
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
            image = self.elastic_transform(image)
            print(image)

        if self.random_crop:
            randomseed = random.randint(0, 100000), random.randint(0, 1000000)
            random_crop = tf.random.uniform(shape=[1], minval=0.6, maxval=1.0)[0]
            original_width = tf.shape(image)[1]
            original_height = tf.cast(tf.shape(image)[0], tf.float32)
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

        if self.do_random_shear:
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
                image = tf.concat([image, image, image], axis=2)
                image = tfa.image.shear_x(image, random_shear, replace=0)
                image, image, image = tf.split(image, 3, axis=2)

        label = image_path[1]
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
            image = tf.image.resize_with_pad(image, self.height, image_width)
        image = 0.5 - image
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, encodedLabel

