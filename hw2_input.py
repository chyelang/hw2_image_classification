"""Routine for decoding the hw2 .jpeg files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import tensorflow as tf
from tensorflow.python.training import queue_runner
import re
import logging
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import configparser
import augmentation

# parse arguments passed by command line by FLAGS
FLAGS = tf.app.flags.FLAGS
section = FLAGS.section
# section = 'lenovo'
config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)
NUM_CLASSES = config.getint(section, 'NUM_CLASSES')
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = config.getint(section, 'NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN')
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = config.getint(section, 'NUM_EXAMPLES_PER_EPOCH_FOR_EVAL')

IMAGE_SIZE_before_random_crop = config.getint(section, 'IMAGE_SIZE_before_random_crop')
# Resize the original images to this size. (the small edge)
IMAGE_SIZE_before_augmentation = config.getint(section, 'IMAGE_SIZE_before_augmentation')
# Process images of this size.
IMAGE_SIZE_to_feed = config.getint(section, 'IMAGE_SIZE_to_feed')


def read_hw2(input_queue):
	"""Reads and parses examples from hw2 data files.

	Args:
	  filename_queue: A queue of strings with the filenames to read from.

	Returns:
	  An object representing a single example, with the following fields:
		height: number of rows in the result
		width: number of columns in the result
		depth: number of color channels in the result
		key: a scalar string Tensor describing the filename & record number
		  for this example.
		label0: an int32 Tensor
		uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class Hw2Record(object):
		pass

	result = Hw2Record()

	label = input_queue[1]
	# label0 = input_queue[1].dequeue()
	result.label = tf.cast(label, tf.int32)
	# result.key, image_file = tf.WholeFileReader().read(input_queue[0])
	image_file = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_file, channels=3)
	# Take width/height
	initial_width = tf.shape(image)[0]
	initial_height = tf.shape(image)[1]

	# NHWC
	# Function for resizing
	def _resize(x, y, mode):
		# Take the greater value, and use it for the ratio
		if mode == "max":
			max_ = tf.maximum(initial_width, initial_height)
			ratio = tf.to_float(max_) / tf.constant(IMAGE_SIZE_before_random_crop, dtype=tf.float32)
		elif mode == "min":
			min_ = tf.minimum(initial_width, initial_height)
			ratio = tf.to_float(min_) / tf.constant(IMAGE_SIZE_before_random_crop, dtype=tf.float32)
		new_width = tf.to_float(initial_width) / ratio
		new_height = tf.to_float(initial_height) / ratio
		return tf.to_int32(new_width), tf.to_int32(new_height)

	with tf.control_dependencies([image]):
		# resize and keeping the initial ratio height/width
		new_w, new_h = _resize(initial_width, initial_height, "min")

		# resize to a square
		# new_w, new_h = IMAGE_SIZE_before_random_crop,IMAGE_SIZE_before_random_crop
		resized_image = tf.image.resize_images(image, [new_w, new_h])
	result.uint8image = tf.cast(resized_image, tf.uint8)
	image = tf.expand_dims(image, 0)
	resized_image = tf.expand_dims(resized_image, 0)
	tf.summary.image('images_before_resize', image)
	tf.summary.image('images_after_resize', resized_image)
	return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	"""Construct a queued batch of images and labels.

	Args:
	  image: 3-D Tensor of [height, width, 3] of type.float32.
	  label: 1-D Tensor of type.int32
	  min_queue_examples: int32, minimum number of samples to retain
		in the queue that provides of batches of examples.
	  batch_size: Number of images per batch.
	  shuffle: boolean indicating whether to use a shuffling queue.

	Returns:
	  images: Images. 4D tensor of [batch_size, height, width, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""
	# Create a queue that shuffles the examples, and then
	# read 'batch_size' images + labels from the example queue.
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size)

	# Display the training images in the visualizer.
	# tf.summary.image('images', images)

	return images, tf.reshape(label_batch, [batch_size])


def wrap_with_queue(tensor, dtypes=tf.string):
	queue = tf.FIFOQueue(100, dtypes=dtypes)
	enqueue_op = queue.enqueue(tensor)
	queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, [enqueue_op]))
	return queue


def distorted_inputs(data_dir, batch_size):
	"""Construct distorted input for hw2 training using the Reader ops.

	Args:
	  data_dir: Path to the hw2 data directory.
	  batch_size: Number of images per batch.

	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""
	image_dirs = os.listdir(data_dir)
	image_paths = []
	labels = []
	for image_dir in image_dirs:
		images = os.listdir(data_dir + '/' + image_dir)
		label = re.sub("\D", "", image_dir)
		for image in images:
			image_paths.append(data_dir + '/' + image_dir + '/' + image)
			labels.append(int(label))

	# Makes an input queue
	image_paths_op, labels_op = tf.train.slice_input_producer([image_paths, labels])

	with tf.name_scope('data_augmentation'):
		read_input = read_hw2([image_paths_op, labels_op])
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE_to_feed
		width = IMAGE_SIZE_to_feed

		# Image processing for training the network. Note the many random
		# distortions applied to the image.

		# Randomly crop a [height, width] section of the image.
		image_size_to_crop = tf.random_uniform((), IMAGE_SIZE_before_augmentation, IMAGE_SIZE_before_random_crop, dtype=tf.int32)
		tf.summary.scalar("image_size_to_crop", image_size_to_crop)
		distorted_image = tf.random_crop(reshaped_image, [image_size_to_crop, image_size_to_crop, 3])

		distorted_image = tf.image.resize_images(distorted_image, [IMAGE_SIZE_before_augmentation, IMAGE_SIZE_before_augmentation])
		tf.summary.image('images_before_augmentation', tf.expand_dims(distorted_image, 0))
		distorted_image = augmentation.image_augmentation(distorted_image)
		tf.summary.image('images_after_augmentation', tf.expand_dims(distorted_image, 0))
		distorted_image = tf.image.resize_images(distorted_image, [height, width])
		tf.summary.image('images_before_standardization', tf.expand_dims(distorted_image, 0))

		# Subtract off the mean and divide by the variance of the pixels.
		float_image = tf.image.per_image_standardization(distorted_image)
		tf.summary.image('images_after_standardization', tf.expand_dims(float_image,0))

		# Set the shapes of tensors.
		float_image.set_shape([height, width, 3])
		# read_input.label0.set_shape([1])

		# Ensure that the random shuffling has good mixing properties.
		min_fraction_of_examples_in_queue = 0.4
		# # The num of examples in each epoch is not the same, but have a low limit?
		min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
								 min_fraction_of_examples_in_queue)
		print('Filling queue with %d hw2 images before starting to train. '
			  'This will take a few minutes.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=True)


def inputs(is_test_eval, data_dir, batch_size):
	"""Construct input for hw2 evaluation using the Reader ops.

	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.
	  data_dir: Path to the CIFAR-10 data directory.
	  batch_size: Number of images per batch.

	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""
	image_dirs = os.listdir(data_dir)
	image_paths = []
	labels = []
	for image_dir in image_dirs:
		images = os.listdir(data_dir + '/' + image_dir)
		label = re.sub("\D", "", image_dir)
		for image in images:
			image_paths.append(data_dir + '/' + image_dir + '/' + image)
			labels.append(int(label))

	# Makes an input queue
	image_paths_op, labels_op = tf.train.slice_input_producer([image_paths, labels])
	if not is_test_eval:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	with tf.name_scope('input'):
		# Read examples from files in the filename queue.
		read_input = read_hw2([image_paths_op, labels_op])
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE_to_feed
		width = IMAGE_SIZE_to_feed

		# Image processing for evaluation.
		# Crop the central [height, width] of the image.
		resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
															   IMAGE_SIZE_before_random_crop,
															   IMAGE_SIZE_before_random_crop)
		tf.summary.image('eval_images_after_central_crop', tf.expand_dims(resized_image, 0))
		resized_image = tf.image.resize_images(resized_image, [height, width])
		tf.summary.image('eval_images_before_standardization', tf.expand_dims(resized_image, 0))

		# Subtract off the mean and divide by the variance of the pixels.
		float_image = tf.image.per_image_standardization(resized_image)
		tf.summary.image('eval_images_after_standardization', tf.expand_dims(resized_image, 0))

		# Set the shapes of tensors.
		float_image.set_shape([height, width, 3])
		# read_input.label0.set_shape([1])

		# Ensure that the random shuffling has good mixing properties.
		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(num_examples_per_epoch *
								 min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=False)


if __name__ == '__main__':
	distorted_inputs("/home/charles/PycharmProjects/hw2_image_classification/data/dset1/train", 5)
