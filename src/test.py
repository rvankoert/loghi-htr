import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.keras.backend.floatx())
# print(tf.keras.backend.set_floatx('float32'))
# print(tf.keras.backend.floatx())

print(tf.keras.backend.set_floatx('float16'))
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
print(tf.keras.backend.floatx())

img_path = tf.keras.utils.get_file('tf_logo.png', 'https://tensorflow.org/images/tf_logo.png')
img_raw = tf.io.read_file(img_path)
img = tf.io.decode_image(img_raw)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, [500,500])

# plt.title("TensorFlow Logo with shape {}".format(img.shape))
# _ = plt.imshow(img)
# _ = plt.imshow(img)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(H.history["loss"], label="train_loss")
# plt.plot(H.history["val_loss"], label="val_loss")
# plt.plot(H.history["accuracy"], label="train_acc")
# plt.plot(H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
#
input_img = tf.image.convert_image_dtype(tf.expand_dims(img, 0), tf.dtypes.float32)
#
flow_shape = [1, input_img.shape[1], input_img.shape[2], 2]
init_flows = np.float32(np.random.normal(size=flow_shape) * 2.0)
dense_img_warp = tfa.image.dense_image_warp(input_img, init_flows)
dense_img_warp = tf.squeeze(dense_img_warp, 0)
_ = plt.imshow(dense_img_warp)
rotate = tfa.image.rotate(img, tf.constant(np.pi/8))
_ = plt.imshow(rotate)

# transform = tfa.image.transform(img, [1.0, 1.0, -250, 0.0, 1.0, 0.0, 0.0, 0.0])
# _ = plt.imshow(transform)
#
# delta = 0.5
# lower_saturation = 0.1
# upper_saturation = 0.9
# lower_value = 0.2
# upper_value = 0.8
# rand_hsvinyiq = tfa.image.random_hsv_in_yiq(img, delta, lower_saturation, upper_saturation, lower_value, upper_value)
# _ = plt.imshow(rand_hsvinyiq)
#
# delta = 0.5
# saturation = 0.3
# value = 0.6
# adj_hsvinyiq = tfa.image.adjust_hsv_in_yiq(img, delta, saturation, value)
# _ = plt.imshow(adj_hsvinyiq)

# gray = tf.image.convert_image_dtype(bw_img,tf.uint8)
# # The op expects a batch of images, so add a batch dimension
# gray = tf.expand_dims(gray, 0)
# eucid = tfa.image.euclidean_dist_transform(gray)
# eucid = tf.squeeze(eucid, (0, -1))
# _ = plt.imshow(eucid, cmap='gray')
#
# transform = tfa.image.transform(img, [1.0, 1.0, -50, 0.0, 1.0, 0.0, 0.0, 0.0])
# _ = plt.imshow(transform)
#
MAX_SHEAR_LEVEL_HORIZONTAL = 1.0
MAX_SHEAR_LEVEL_VERTICAL = 0.0
#
transform =  tfa.image.transform(img, [1.0, MAX_SHEAR_LEVEL_HORIZONTAL * tf.random.uniform(shape=[],minval=-1,maxval=1), 0.0, MAX_SHEAR_LEVEL_VERTICAL * tf.random.uniform(shape=[],minval=-1,maxval=1), 1.0, 0.0, 0.0, 0.0])

# transform = tfa.image.translate(img, [HSHIFT * tf.random.uniform(shape=[], minval=-1, maxval=1),
#                                 VSHIFT * tf.random.uniform(shape=[], minval=-1,
#                                                            maxval=1)])  # [dx dy] shift/translation

_ = plt.imshow(transform)


plt.savefig('/tmp/plt.png')
