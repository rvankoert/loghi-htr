# Imports

# > Standard Library
import random
import logging

import keras
# > Local dependencies
from setup.arg_parser import get_args

# > Third party libraries
import tensorflow as tf
import elasticdeform.tf as etf
import tensorflow_models as tfm
from skimage.filters import threshold_otsu, threshold_sauvola
import numpy as np
import matplotlib.pyplot as plt


class ShearXLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply shear transformation along the x-axis to the input image.

        Parameters
        ----------
        - image (numpy.ndarray): Input image, can be 3-channel (RGB) or
          4-channel (RGBA).

        Returns
        -------
        - numpy.ndarray: Sheared image.

        Example
        -------
        >>> input_image = np.random.rand(256, 256, 3)
        >>> sheared_result = shear_x(input_image, 0.5)
        """

        # Calculate random shear_factor
        shear_factor = tf.random.uniform(shape=[1], minval=-1.0, maxval=1.0)[0]

        # Define the shear matrix
        shear_matrix = [1.0, shear_factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # Get the dynamic shape of the input tensor
        input_shape = tf.shape(inputs)

        # Ensure the output shape is the same as input shape for height and width
        output_shape = [input_shape[1], input_shape[2]]

        # Flatten the shear matrix for batch processing
        shear_matrix_tf = tf.reshape(tf.convert_to_tensor(shear_matrix,
                                                          dtype=tf.dtypes.float32),
                                     [1, 8])

        # Apply the shear transformation
        sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=inputs,
            transforms=shear_matrix_tf,
            output_shape=output_shape,
            fill_value=0,
            fill_mode="CONSTANT",
            interpolation="NEAREST")

        return sheared_image


class ElasticTransformLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Apply elastic transformation to an image.

        Parameters
        ----------
        - original (tf.Tensor): The original image to be transformed.

        Returns
        -------
        - tf.Tensor: The image after elastic transformation.

        Notes
        -----
        - Elastic transformation introduces local deformations to the image to
          enhance robustness and variability in the dataset.
        - It uses a random displacement field generated from a normal
          distribution.
        - The `etf.deform_grid` function is employed for the deformation,
          allowing control over the axis and interpolation order.

        Example
        -------
        ```python
        transformer = ImageTransformer()
        original_image = load_image("path/to/image.jpg")
        transformed_image = transformer.elastic_transform(original_image)
        ```
        """
        displacement_val = tf.random.normal([2, 3, 3]) * 5
        # Check highest value in input
        max_val = tf.math.reduce_max(inputs)
        # Interpolation-order heavily influences result
        x_deformed = etf.deform_grid(
            inputs, displacement_val, axis=(0, 1), order=1,
            cval=tf.get_static_value(max_val)
        )
        return x_deformed


class DistortImageLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Distorts the input image based on specified parameters.

        Parameters:
        - image (tf.Tensor): Input image as a TensorFlow tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Distorted image as a TensorFlow tensor.
        """
        print("before distort shaep: ", inputs.shape)
        image_width = tf.shape(inputs)[1]
        image_height = tf.shape(inputs)[0]
        if inputs.shape[-1] == 4:
            # Workaround for a bug in shear_x where alpha causes errors
            channel1, channel2, channel3, alpha = tf.split(inputs, 4, axis=2)
            image = tf.concat([channel1, channel2, channel3], axis=2)

            # Apply random JPEG quality to the image
            image = tf.image.random_jpeg_quality(image, 50, 100)

            # Split the channels again after distortion
            channel1, channel2, channel3 = tf.split(image, 3, axis=2)

            # Concatenate the channels along with alpha channel
            image = tf.concat([channel1, channel2, channel3, alpha], axis=2)

            # Clean up variables
            del channel1, channel2, channel3, alpha
        else:
            # Apply random JPEG quality to the image
            image = tf.image.random_jpeg_quality(inputs, 20, 100)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # print(image)
        # image = tf.image.resize(image, [image_height, image_width])
        print("after distort shape: ", image.shape)

        return image


class RandomVerticalCropLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomVerticalCropLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Applies random crop to each image in the input batch.

        Parameters:
        - inputs (tf.Tensor): Input batch of images as a TensorFlow tensor.

        Returns:
        - tf.Tensor: Batch of images after applying random crop.
        """
        # Get the dtype of the input tensor
        input_dtype = inputs.dtype

        # Get the original height and width of the images
        height, width, channels = tf.unstack(tf.shape(inputs)[1:])

        # Generate a random crop factor for each image in the batch
        min_crop_factor = tf.constant(0.6, dtype=tf.float32)
        crop_height = tf.cast(min_crop_factor * tf.cast(height,
                                                        tf.float32), tf.int32)

        # Define the crop size
        crop_size = (crop_height, width, channels)

        # Crop each image in the batch
        def crop_image(image):
            cropped = tf.image.random_crop(image, crop_size)
            # Cast back to the original dtype
            return tf.cast(cropped, input_dtype)

        # Ensure the output dtype matches the input
        cropped_images = tf.map_fn(crop_image, inputs, dtype=input_dtype)

        return cropped_images


class ResizeWithPadLayer(tf.keras.layers.Layer):
    def __init__(self, additional_width=50, **kwargs):
        super(ResizeWithPadLayer, self).__init__(**kwargs)
        self.additional_width = additional_width

    def call(self, inputs):
        """
        Resize and pad the input images.

        Parameters:
        - inputs (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Resized and padded image tensor.
        """
        new_width = tf.shape(inputs)[2] + self.additional_width
        print("PRE PADDING SHAPE ", inputs.shape)
        padded_img = tf.image.resize_with_pad(inputs, tf.shape(inputs)[1], new_width)
        print("POST PADDING SHAPE ", padded_img.shape)
        return padded_img


class BinarizeLayer(tf.keras.layers.Layer):
    def __init__(self, method='otsu', window_size=51, **kwargs):
        super(BinarizeLayer, self).__init__(**kwargs)
        self.method = method
        self.window_size = window_size

    def call(self, inputs):
        """
        Binarize the input tensor using either Otsu's or Sauvola's thresholding.

        Parameters:
        - inputs (tf.Tensor): Input image tensor.
        - method (str): Thresholding method ('otsu' or 'sauvola').

        Returns:
        - tf.Tensor: Binarized image tensor.
        """

        # Handle different channels
        # channels = inputs.shape[-1]
        # if channels == 1:
        #     processed_input = inputs
        # elif channels in [3, 4]:
        #     processed_input = tf.image.rgb_to_grayscale(inputs[:, :, :3])
        # else:
        #     raise NotImplementedError(
        #         "Unsupported number of channels. Supported values are 1, 3, or 4.")

        # Convert tensor to float32 for thresholding
        processed_input = tf.cast(inputs, tf.float32)

        # Apply the selected thresholding method
        if self.method == 'otsu':
            def otsu_thresh(image):
                thresh = threshold_otsu(image)
                return ((image > thresh) * 1).astype(np.float32)

            binarized = tf.numpy_function(otsu_thresh, [processed_input],
                                          tf.float32)

        elif self.method == 'sauvola':
            def sauvola_thresh(image):
                thresh = threshold_sauvola(image, window_size=self.window_size)
                return ((image > thresh) * 1).astype(np.float32)

            binarized = tf.numpy_function(sauvola_thresh, [processed_input],
                                          tf.float32)

        else:
            raise ValueError(
                "Invalid method. Supported methods are 'otsu' and 'sauvola'.")

        # Ensure output shape is set correctly after numpy_function
        binarized.set_shape(processed_input.shape)

        # Convert the binary image to float32
        return tf.cast(binarized, tf.float32)


class InvertImageLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Invert the pixel values of the input tensor.

        Parameters:
        - tensor (tf.Tensor): Input image tensor.
        - channels (int): Number of channels in the image.

        Returns:
        - tf.Tensor: Inverted image tensor.
        """
        # Get the channels from the input
        channels = inputs.shape[-1]

        if (str(inputs.numpy().dtype).startswith("uint") or
                str(inputs.numpy().dtype).startswith("int")):
            max_value = 255
        else:
            max_value = 1

        if channels == 4:
            channel1, channel2, channel3, alpha = tf.split(inputs, 4, axis=2)
            channel1 = tf.convert_to_tensor(max_value - channel1.numpy())
            channel2 = tf.convert_to_tensor(max_value - channel2.numpy())
            channel3 = tf.convert_to_tensor(max_value - channel3.numpy())

            return tf.concat([channel1, channel2, channel3, alpha], axis=2)

        else:
            return tf.convert_to_tensor(max_value - inputs.numpy())


class BlurImageLayer(tf.keras.layers.Layer):

    def __init__(self, mild_blur=False, **kwargs):
        super(BlurImageLayer, self).__init__(**kwargs)
        self.mild_blur = mild_blur

    def call(self, inputs):
        """
        Apply random Gaussian blur to the input tensor

        Parameters:
        - image (tf.Tensor): Input image tensor.

        Returns:
        - tf.Tensor: Blurred image tensor.
        """
        # Get the dynamic shape of the input tensor
        input_shape = tf.shape(inputs)

        # Check if the input tensor is large enough for the desired blur
        min_dimension = tf.minimum(input_shape[1], input_shape[2])
        if min_dimension < 10:  # Example threshold, adjust as needed
            # Skip the blur or apply a milder blur if the image is too small
            return inputs


        if self.mild_blur:
            blur_factor = 1
        else:
            blur_factor = round(random.uniform(0.1, 2), 1)
        return tfm.vision.augment.gaussian_filter2d(inputs,
                                                    filter_shape=(10, 10),
                                                    sigma=blur_factor)


class RandomWidthLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        """
        Randomly adjusts the width of the input image.

        Parameters:
        - image (tf.Tensor): Input image as a TensorFlow tensor.

        Returns:
        - tf.Tensor: Processed image as a TensorFlow tensor.
        """

        # Get the width and height of the input image
        image_width = tf.shape(inputs)[1]
        image_height = tf.shape(inputs)[0]
        channels = tf.shape(inputs)[2]

        # Generate a random width scaling factor between 0.75 and 1.25
        random_width = tf.random.uniform(shape=[1], minval=0.75, maxval=1.25)[0]

        # Scale the width of the image by the random factor
        random_width *= float(image_width)
        image_width = int(random_width)

        # Convert the image to float32 dtype
        image = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

        # Resize the image to the new width while maintaining the original height
        # image = tf.reshape(image, (image_height, image_width, channels))

        return image


def inspect_augments(image, augment_model, file_name="augment_test.png"):
    fig, ax = plt.subplots(nrows=10, figsize=(16, 10))
    fig.suptitle('Image augments:', fontsize=16)
    for i in range(10):
        augmented_image = augment_model(image)
        ax[i].imshow(augmented_image, cmap='gray')
        ax[i].set_title(str([layer.name for layer in augment_model.layers]))
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def generate_random_augment_list(augment_options):
    # Ensure there are at least 2 elements in the list
    num_rand_augments = random.randint(2, len(augment_options))

    # Randomly select unique elements from the input list
    augment_layers = random.sample(augment_options, num_rand_augments)

    # If invert_image and binarize otsu is done -> remove te invert_image
    if BinarizeLayer and (InvertImageLayer() in augment_layers):
        augment_layers = augment_layers.remove(InvertImageLayer())

    return augment_layers


def get_augment_classes():
    # Retrieve a list of possible augmentations
    g = globals().copy()
    augment_options = []
    for name, obj in g.items():
        # EXCLUDE INVERT FOR NOW WHILE FIXING
        if name.startswith("Invert"):
            continue
        if isinstance(obj, type):
            layer_instance = obj()
            augment_options.append(layer_instance)
    return augment_options


def get_augment_model():
    logger = logging.getLogger(__name__)
    args = get_args()
    augment_selection = []

    if args.aug_distort_jpeg:
        logger.info("Data augment: distort_jpeg")
        augment_selection.append(DistortImageLayer())

    if args.aug_elastic_transform:
        logger.info("Data augment: elastic_transform")
        augment_selection.append(ElasticTransformLayer())

    if args.aug_random_crop:
        logger.info("Data augment: random_crop")
        augment_selection.append(RandomVerticalCropLayer())

    if args.aug_random_width:
        logger.info("Data augment: random_width")
        augment_selection.append(RandomWidthLayer())

    # ADD PaddingLayer below v
    # image = tf.image.resize_with_pad(image,
    #                                  self.height, tf.shape(image)[1] + 50)
    # Add padding to ensure text line does not fall of the image

    # augment_selection.append(ResizeWithPadLayer())



    if args.aug_random_shear:
        logger.info("Data augment: shear_x")
        augment_selection.append(ShearXLayer())

    if args.aug_binarize_sauvola:
        logger.info("Data augment: binarize_sauvola")
        augment_selection.append(BinarizeLayer(method='sauvola',
                                               window_size=51))
    if args.aug_binarize_otsu:
        logger.info("Data augment: binarize_otsu")
        augment_selection.append(BinarizeLayer(method="otsu"))

    if args.aug_blur:
        logger.info("Data augment: blur_image")
        augment_selection.append(BlurImageLayer())

    if args.aug_invert:
        logger.info("Data augment: aug_invert")
        augment_selection.append(InvertImageLayer())

    return augment_selection


def make_augment_model(augment_options, augment_selection=None):
    # Take random augments from all possible augment options
    if augment_selection is None:
        augment_selection = generate_random_augment_list(augment_options)

    # Check if both types of binarization are present, remove Sauvola if true
    otsu_present = any(isinstance(obj, BinarizeLayer)
                       and obj.method == 'otsu' for obj in augment_selection)
    sauvola_present = any(isinstance(obj, BinarizeLayer)
                          and obj.method == 'sauvola' for obj in
                          augment_selection)

    if otsu_present and sauvola_present:
        # Remove the object of type BinarizeLayer with Sauvola method
        augment_selection = [obj for obj in augment_selection
                             if not (isinstance(obj, BinarizeLayer)
                                     and obj.method == 'sauvola')]

    # Init blur params
    mild_blur = False
    blur_index = -1

    # Check if a blur occurs before binarization, if so replace with mild blur
    for i, augment in enumerate(augment_selection):
        if isinstance(augment, BlurImageLayer):
            mild_blur = True
            blur_index = i
        elif mild_blur and isinstance(augment, BinarizeLayer):
            augment_selection[blur_index] = BlurImageLayer(mild_blur=True)

    # Check if blur/distort occurs after binarization, if so move it to before
    binarize = False
    binarize_index = -1
    for i, augment in enumerate(augment_selection):
        if isinstance(augment, BinarizeLayer):
            binarize = True
            binarize_index = i
        elif binarize and (isinstance(augment, BlurImageLayer)):
            # Remove BlurImageLayer that occurs after binarization
            augment_selection.remove(augment)
            # Insert a mild blur object right before binarize
            augment_selection.insert(binarize_index,
                                     BlurImageLayer(mild_blur=True))
        elif binarize and (isinstance(augment, DistortImageLayer)):
            # Remove DistortImageLayer that occurs after binarization
            augment_selection.remove(augment)
            # Insert the Distort layer before the binarization
            augment_selection.insert(binarize_index, augment)

    print("changed aug selection list = ",
          tf.keras.Sequential(augment_selection).layers)

    adjusted_aug_model = tf.keras.Sequential(augment_selection,
                                             name="data_augment_model")
    return adjusted_aug_model
