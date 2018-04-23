"""Routine for decoding the hw2 .jpeg files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops

import os

curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import tensorflow as tf
from tensorflow.python.training import queue_runner
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Resize the original images to this size. (the small edge)
IMAGE_RESIZE = 160
# Process images of this size.
IMAGE_SIZE = 120
import configparser

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
		label: an int32 Tensor
		uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class Hw2Record(object):
		pass

	result = Hw2Record()

	label = input_queue[1]
	# label = input_queue[1].dequeue()
	result.label = tf.cast(label, tf.int32)
	# result.key, image_file = tf.WholeFileReader().read(input_queue[0])
	image_file = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_file, channels=3)
	# Take width/height
	initial_width = tf.shape(image)[0]
	initial_height = tf.shape(image)[1]

	# NWHC

	# Function for resizing
	def _resize(x, y, mode):
		# Take the greater value, and use it for the ratio
		if mode == "max":
			max_ = tf.maximum(initial_width, initial_height)
			ratio = tf.to_float(max_) / tf.constant(IMAGE_RESIZE, dtype=tf.float32)
		elif mode == "min":
			min_ = tf.minimum(initial_width, initial_height)
			ratio = tf.to_float(min_) / tf.constant(IMAGE_RESIZE, dtype=tf.float32)
		new_width = tf.to_float(initial_width) / ratio
		new_height = tf.to_float(initial_height) / ratio
		return tf.to_int32(new_width), tf.to_int32(new_height)

	with tf.control_dependencies([image]):
		new_w, new_h = _resize(initial_width, initial_height, "min")
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
	num_preprocess_threads = 4
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
	tf.summary.image('images', images)

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

	# image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
	# labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), dtype=tf.int32)
	# # Makes an input queue
	# image_paths_op, labels_op = tf.train.slice_input_producer([image_paths, labels])

	# Makes an input queue
	image_paths_op, labels_op = tf.train.slice_input_producer([image_paths, labels])
	# image_paths_queue = wrap_with_queue(image_paths_op)
	# labels_queue = wrap_with_queue(labels_op, dtypes=tf.int32)

	with tf.name_scope('data_augmentation'):
		# Read examples from files in the filename queue.
		# read_input = read_hw2([image_paths_queue, labels_queue])
		read_input = read_hw2([image_paths_op, labels_op])
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE
		width = IMAGE_SIZE

		# Image processing for training the network. Note the many random
		# distortions applied to the image.

		# Randomly crop a [height, width] section of the image.
		distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

		# Randomly flip the image horizontally.
		distorted_image = tf.image.random_flip_left_right(distorted_image)

		# Because these operations are not commutative, consider randomizing
		# the order their operation.
		# NOTE: since per_image_standardization zeros the mean and makes
		# the stddev unit, this likely has no effect see tensorflow#1458.
		distorted_image = tf.image.random_brightness(distorted_image,
													 max_delta=63)
		distorted_image = tf.image.random_contrast(distorted_image,
												   lower=0.2, upper=1.8)

		# Subtract off the mean and divide by the variance of the pixels.
		float_image = tf.image.per_image_standardization(distorted_image)

		# Set the shapes of tensors.
		float_image.set_shape([height, width, 3])
		# read_input.label.set_shape([1])

		# Ensure that the random shuffling has good mixing properties.
		min_fraction_of_examples_in_queue = 0.4
		# # The num of examples in each epoch is not the same, but have a low limit?
		min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
								 min_fraction_of_examples_in_queue)
		print('Filling queue with %d hw2 images before starting to train. '
			  'This will take a few minutes.' % min_queue_examples)
	# #for debugging
	# tmp = _generate_image_and_label_batch(float_image, read_input.label,
	# 									   min_queue_examples, batch_size,
	# 							  shuffle=False)
	# with tf.Session() as sess:
	# 	sess.run(tf.initialize_all_variables())
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	# 	# for qr in ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS):
	# 	# 	print(qr)
	#
	# 	print(sess.run([image_paths_op, labels_op]))
	# 	# print(sess.run([image_paths_queue.dequeue()]))
	# 	print(sess.run([tmp]))
	# 	coord.join(threads)
	# 	# print(sess.run([image_batch]))
	#
	# 	# for i in range(3):
	# 	# 	sess.run(labels_queue)
	# 	# 	print(labels_queue.dequeue().eval())
	#
	# 	coord.request_stop()
	# 	coord.join(threads)
	# 	sess.close()

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label,
										   min_queue_examples, batch_size,
										   shuffle=True)


def inputs(eval_data, data_dir, batch_size):
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

	image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
	labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), dtype=tf.int32)
	# Makes an input queue
	image_paths_op, labels_op = tf.train.slice_input_producer([image_paths, labels])
	image_paths_queue = wrap_with_queue(image_paths_op)
	labels_queue = wrap_with_queue(labels_op, dtypes=tf.int32)
	if not eval_data:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	with tf.name_scope('input'):
		# Read examples from files in the filename queue.
		read_input = read_hw2([image_paths_queue, labels_queue])
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE
		width = IMAGE_SIZE

		# Image processing for evaluation.
		# Crop the central [height, width] of the image.
		resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
															   height, width)

		# Subtract off the mean and divide by the variance of the pixels.
		float_image = tf.image.per_image_standardization(resized_image)

		# Set the shapes of tensors.
		float_image.set_shape([height, width, 3])
		read_input.label.set_shape([1])

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
