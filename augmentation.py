import tensorflow as tf
import numpy as np
from math import ceil, floor


def get_translate_parameters(index, image_dim, gap):
	if index == 0:
		offset = np.array([0.0, gap], dtype=np.float32)
		size = np.array([image_dim, ceil((1-gap) * image_dim)], dtype=np.int32)
		w_start = 0
		w_end = int(ceil((1-gap) * image_dim))
		h_start = 0
		h_end = image_dim
	elif index == 1: 
		offset = np.array([0.0, -gap], dtype=np.float32)
		size = np.array([image_dim, ceil((1-gap) * image_dim)], dtype=np.int32)
		w_start = int(floor((1 - (1-gap)) * image_dim))
		w_end = image_dim
		h_start = 0
		h_end = image_dim
	elif index == 2:  
		offset = np.array([gap, 0.0], dtype=np.float32)
		size = np.array([ceil((1-gap) * image_dim), image_dim], dtype=np.int32)
		w_start = 0
		w_end = image_dim
		h_start = 0
		h_end = int(ceil((1-gap) * image_dim))
	elif index == 3:
		offset = np.array([-gap, 0.0], dtype=np.float32)
		size = np.array([ceil((1-gap) * image_dim), image_dim], dtype=np.int32)
		w_start = 0
		w_end = image_dim
		h_start = int(floor((1 - (1-gap)) * image_dim))
		h_end = image_dim
	else:  
		offset = np.array([0.0, 0.0], dtype=np.float32)
		size = np.array([image_dim, image_dim], dtype=np.int32)
		w_start = 0
		w_end = image_dim
		h_start = 0
		h_end = image_dim
	return offset, size, w_start, w_end, h_start, h_end


def image_augmentation(image):
	image_dim = image.get_shape().as_list()[0]
	image = tf.expand_dims(image, 0)

	# # randomly scale image
	# scale = tf.random_uniform((), 0.9, 1, dtype=tf.float32)
	# x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
	# x2 = y2 = 0.5 + 0.5 * scale
	# tf.summary.scalar("scale", scale)
	# boxes = tf.Variable([y1, x1, y2, x2], trainable=False, dtype=np.float32)
	# tf.assign(boxes, [y1, x1, y2, x2])
	# boxes = tf.expand_dims(boxes, axis=0)
	# box_ind = tf.zeros((1), dtype=tf.int32)
	# crop_size = tf.constant([image_dim, image_dim], dtype=np.int32)
	# image = tf.image.crop_and_resize(image, boxes, box_ind, crop_size)

	# # randomly glimpse
	# # has bug to fix
	# # init_values = np.ones([1, image_dim, image_dim, 3])
	# init_values = tf.constant(1.0, shape=(1, image_dim, image_dim, 3))
	# image_translated = tf.Variable(init_values, trainable=False, dtype=tf.float32)
	# seed = tf.random_uniform((), 0, 6, dtype=tf.int32)
	# offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(seed, image_dim, 0.10)
	# offset = np.expand_dims(offset, 0)
	# glimpse = tf.image.extract_glimpse(image, size, offset)
	# tf.assign(image_translated, init_values)
	# image = image_translated[:, h_start: h_start + size[0], w_start: w_start + size[1], :].assign(glimpse)

	# # Rotation (at finer angles)
	# degrees_angle = tf.random_uniform((), 0, 360, dtype=tf.float32)
	degrees_angle = tf.random_uniform((), -10, 10, dtype=tf.float32)
	tf.summary.scalar("rotate_angle", degrees_angle)
	radian_value = tf.multiply(degrees_angle, tf.constant(3.14, dtype=tf.float32)) / 180   # Convert to radian
	image = tf.contrib.image.rotate(image, radian_value)

	# add_salt_pepper_noise
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

	image = tf.reduce_sum(image, 0)
	# # Rotation (at 90 degrees)
	# seed = random.randint(0, 3)
	# image = tf.image.rot90(image, k=seed)

	# Randomly flip the image horizontally and vertically.
	image = tf.image.random_flip_left_right(image)
	# image = tf.image.random_flip_up_down(image)

	# brightness and contrast
	image = tf.image.random_brightness(image, max_delta=63)
	image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

	return image
