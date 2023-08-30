from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import tensorflow as tf


def noisy(image):
	row,col = image.shape
	s_vs_p = 0.5
	amount = 0.30
	out = np.copy(image)
	

	# Multi mode
	for x in range(10):
		num_salt = np.ceil(amount * image.size / 10)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		for i in image.shape]
		out[tuple(coords)] = np.random.randint(0,255)


#	# Salt mode
#	num_salt = np.ceil(amount * image.size * s_vs_p)
#	coords = [np.random.randint(0, i - 1, int(num_salt))
#	for i in image.shape]
#	out[coords] = 255
#
 #     # Pepper mode
#	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
#	coords = [np.random.randint(0, i - 1, int(num_pepper))
#	for i in image.shape]
#	out[coords] = 0
	return out

def preprocess(img, imgSize, dataAugmentation=False):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"

	# there are damaged files in IAM dataset - just use black image instead
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# increase dataset size by applying random stretches to the images
	if dataAugmentation:
		stretch = (random.random() - 0.5) # -0.5 .. +0.5
		wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
		img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5

		rows,cols = img.shape
		channels = 1
		shift = int((random.random()-0.5)*rows)
		if (shift>=0):
			pts1 = np.float32([[0,0],[cols,0],[cols,rows]])
			pts2 = np.float32([[0,0],[cols,0],[cols+shift,rows]])
		else:
			pts1 = np.float32([[0,0],[cols,0],[cols,rows]])
			pts2 = np.float32([[abs(shift),0],[cols+abs(shift),0],[cols,rows]])

		M = cv2.getAffineTransform(pts1,pts2)
		dst = np.ones([cols+abs(shift), rows]) * 255
		img = cv2.warpAffine(img,M,(cols+abs(shift),rows), dst, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

	
	# create target image and copy sample image into it
	(wt, ht, channels) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img

	xoffset, yoffset = newSize
#	xoffset = int(random.random() * (wt-xoffset))
#	yoffset = int(random.random() * (ht-yoffset))
	target = np.ones([ht, wt]) * 255
#	yoffset = 0
#	print (yoffset)
	yoffset = int(0.5 * (ht-yoffset))
#	print (ht)
#	print (yoffset)
	xoffset = int((wt-xoffset)/2) # center in the middle
	xoffset = 0
#	yoffset = 0
	if dataAugmentation:
		img = noisy(img)
		target[yoffset:newSize[1]+yoffset, xoffset:newSize[0]+xoffset] = img
#		target[0:newSize[1], 0:newSize[0]] = img
	else:
#		xoffset, yoffset = newSize
#		xoffset = int( (wt-xoffset)/2)
#		yoffset = int((ht-yoffset)/2)

#		target[yoffset:newSize[1]+yoffset, xoffset:newSize[0]+xoffset] = img
		target[0:newSize[1], 0:newSize[0]] = img
	# transpose for TF
	target = cv2.transpose(target)
#	target = tf.transpose(target, perm=[1, 0])
	print (target.shape)
	return target

