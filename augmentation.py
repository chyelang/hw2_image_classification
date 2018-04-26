import tensorflow as tf
import numpy as np
import random
from math import ceil, floor


def get_translate_parameters(index, image_dim):
	if index == 0:  # Translate left 20 percent
		offset = np.array([0.0, 0.2], dtype=np.float32)
		size = np.array([image_dim, ceil(0.8 * image_dim)], dtype=np.int32)
		w_start = 0
		w_end = int(ceil(0.8 * image_dim))
		h_start = 0
		h_end = image_dim
	elif index == 1:  # Translate right 20 percent
		offset = np.array([0.0, -0.2], dtype=np.float32)
		size = np.array([image_dim, ceil(0.8 * image_dim)], dtype=np.int32)
		w_start = int(floor((1 - 0.8) * image_dim))
		w_end = image_dim
		h_start = 0
		h_end = image_dim
	elif index == 2:  # Translate top 20 percent
		offset = np.array([0.2, 0.0], dtype=np.float32)
		size = np.array([ceil(0.8 * image_dim), image_dim], dtype=np.int32)
		w_start = 0
		w_end = image_dim
		h_start = 0
		h_end = int(ceil(0.8 * image_dim))
	else:  # Translate bottom 20 percent
		offset = np.array([-0.2, 0.0], dtype=np.float32)
		size = np.array([ceil(0.8 * image_dim), image_dim], dtype=np.int32)
		w_start = 0
		w_end = image_dim
		h_start = int(floor((1 - 0.8) * image_dim))
		h_end = image_dim
	return offset, size, w_start, w_end, h_start, h_end


def image_augmentation(image):
	image_dim = image.get_shape().as_list()[0]

	# randomly scale image
	scale = random.uniform(0.7, 1)
	x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
	x2 = y2 = 0.5 + 0.5 * scale
	boxes = np.array([y1, x1, y2, x2], dtype=np.float32)
	boxes = np.expand_dims(boxes, axis=0)
	box_ind = np.zeros((1), dtype=np.int32)
	crop_size = np.array([image_dim, image_dim], dtype=np.int32)
	image = tf.expand_dims(image, 0)
	image = tf.image.crop_and_resize(image, boxes, box_ind, crop_size)

	# # randomly glimpse
	# init_values = np.ones([1, image_dim, image_dim, 3])
	# image_translated = tf.Variable(init_values, trainable=False, dtype=np.float32)
	# seed = random.randint(0, 3)
	# offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(seed, image_dim)
	# offset = np.expand_dims(offset,0)
	# glimpse = tf.image.extract_glimpse(image, size, offset)
	# image = image_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :].assign(glimpse)

	# # Rotation (at 90 degrees)
	# seed = random.randint(0, 3)
	# image = tf.image.rot90(image, k=seed)

	# Rotation (at finer angles)
	degrees_angle = random.randint(0, 360)
	radian_value = degrees_angle * 3.14159 / 180  # Convert to radian
	image = tf.contrib.image.rotate(image, radian_value)

	# add_salt_pepper_noise
	image = tf.reduce_sum(image, 0)
	# salt_vs_pepper = 0.2
	# amount = 0.04
	# num_salt = np.ceil(amount * image_dim * salt_vs_pepper)
	# num_pepper = np.ceil(amount * image_dim * (1.0 - salt_vs_pepper))
	# # Add Salt noise
	# coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.get_shape().as_list()]
	# tmp =  list(zip(coords[0], coords[1], coords[2]))
	# image = image[index[0],index[1],index[2]].assign(1.0)
	# # Add Pepper noise
	# coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.get_shape().as_list()]
	# image = image[coords[0], coords[1], :].assign(0)

	# Randomly flip the image horizontally and vertically.
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)

	# brightness and contrast
	# image = tf.image.random_brightness(image, max_delta=63)
	# image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

	return image
