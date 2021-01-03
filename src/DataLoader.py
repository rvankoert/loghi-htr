from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from tensorflow.keras import layers
import tensorflow as tf


class DataLoader:

	dataAugmentation = False
	currIdx = 0
	charList = []
	samples = []
	validation_dataset=[];

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []

		f=open('/scratch/train_data/linestrips/all.txt')
		chars = set()
		bad_samples = []
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			fileName = lineSplit[0]

			# GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen).replace("|"," ")
			# chars = chars.union(set(list(gtText)))
			chars = chars.union(set(char for label in gtText for char in label))
			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue

			# put sample into list
			self.samples.append((gtText, fileName))


		# some images in the IAM dataset are known to be damaged, don't show warning for them
		if len(bad_samples)>0:
			print("Warning, damaged images found:", bad_samples)

		# split into training and validation set: 95% - 5%
		random.seed(42)
		# random.shuffle(self.samples)

		# number of randomly chosen samples per epoch for training
		self.numTrainSamplesPerEpoch = 64
		
		# list of all chars in dataset
		self.charList = sorted(list(chars))

		self.char_to_num = layers.experimental.preprocessing.StringLookup(
			vocabulary=list(self.charList), num_oov_indices=0, mask_token=None
		)
		# Mapping integers back to original characters
		self.num_to_char = layers.experimental.preprocessing.StringLookup(
			vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
		)

	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	def encode_single_sample(self,img_path, label):
		# 1. Read image
		img = tf.io.read_file(img_path)
		# 2. Decode and convert to grayscale
		img = tf.io.decode_png(img, channels=1)
		# 3. Convert to float32 in [0, 1] range
		img = tf.image.convert_image_dtype(img, tf.float32)
		# 4. Resize to the desired size
		img = tf.image.resize_with_pad(img, 64, 1024)
		# 5. Transpose the image because we want the time
		# dimension to correspond to the width of the image.
		img = tf.transpose(img, perm=[1, 0, 2])
		# 6. Map the characters in label to numbers
		label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
		# 7. Return a dict as our model is expecting two inputs
		return {"image": img, "label": label}

	def split_data(self, images, labels, train_size=0.9, shuffle=True):
		# 1. Get the total size of the dataset
		size = len(images)
		# 2. Make an indices array and shuffle it, if required
		indices = np.arange(size)
		if shuffle:
			np.random.shuffle(indices)
		# 3. Get the size of training samples
		train_samples = int(size * train_size)
		# 4. Split data into training and validation sets
		x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
		x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
		return x_train, x_valid, y_train, y_valid


	def getValidationDataSet(self):
		return self.validation_dataset

	def getTrainDataSet(self):
		"iterator"

		#		label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

		#gtTexts = [char_to_num(tf.strings.unicode_split(self.samples[i].gtText, input_encoding="UTF-8")) for i in batchRange]
#		gtTexts = [char_to_num(tf.strings.unicode_split(self.samples[i].gtText, input_encoding="UTF-8")) for i in batchRange]
		# gtTexts = [char_to_num(tf.strings.unicode_split("test", input_encoding="UTF-8")) for i in batchRange]
		gtTexts = [self.samples[i][0].ljust(128,' ') for i in range(len(self.samples))]
		# gtTexts = ["test test " for i in range(len(self.samples))]

		max_length = max([len(label) for label in gtTexts])
		print("max_length: "+ str(max_length))
		# gtTexts = [(str(i)+"a123456").ljust(10,'\0') for i in batchRange]

#		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_UNCHANGED), self.imgSize, self.dataAugmentation) for i in batchRange]
		imgs = [self.samples[i][1] for i in range(len(self.samples))]

		x_train, x_valid, y_train, y_valid = self.split_data(np.array(imgs),np.array(gtTexts))
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

		train_dataset = (
			train_dataset.map(
				self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
			)
			.batch(self.batchSize)
			.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		)

		validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
		print (validation_dataset)
		self.validation_dataset = (
			validation_dataset.map(
				self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
			)
				.batch(self.batchSize)
				.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		)

		return train_dataset