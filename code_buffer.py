# import tensorflow as tf
# import glob
# from tensorflow.python.training import queue_runner
#
# def wrap_with_queue(tensor, dtypes=tf.string):
#     queue = tf.FIFOQueue(1, dtypes=dtypes)
#     enqueue_op = queue.enqueue(tensor)
#     queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, [enqueue_op]))
#     return queue
#
# image_files = glob(...)
# label_files = [s.replace('image.jpg', 'labels.jpg') for s in image_files]
# # Create a queue where the elements are pairs (image, label)
# image_file_op, label_file_op = tf.train.slice_input_producer([image_files, label_files])
# image_queue = wrap_with_queue(image_file_op)
# label_queue = wrap_with_queue(label_file_op)
#
# images, labels = [], []
# for _ in num_threads:
#     # Create two readers for each thread - one for the image and one for the label
#     image_reader, label_reader = tf.SomeReader(), tf.SomeReader()
#     # Read the image and label pair
#     _, image_op = image_reader.read(image_queue)
#     _, label_op = label_reader.read(label_queue)
#     images.append(image_op)
#     labels.append(label_op)
#
# sess = tf.Session()
# tf.train.start_queue_runners(sess=sess)

from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
for x in local_device_protos:
	print(x.name)

import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.training import session_run_hook


class EarlyStoppingHook(session_run_hook.SessionRunHook):
	"""Hook that requests stop at a specified step."""

	def __init__(self, monitor='val_loss', min_delta=0, patience=0,
				 mode='auto'):
		"""
		"""
		self.monitor = monitor
		self.patience = patience
		self.min_delta = min_delta
		self.wait = 0
		if mode not in ['auto', 'min', 'max']:
			logging.warning('EarlyStopping mode %s is unknown, '
							'fallback to auto mode.', mode, RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
		elif mode == 'max':
			self.monitor_op = np.greater
		else:
			if 'acc' in self.monitor:
				self.monitor_op = np.greater
			else:
				self.monitor_op = np.less

		if self.monitor_op == np.greater:
			self.min_delta *= 1
		else:
			self.min_delta *= -1

		self.best = np.Inf if self.monitor_op == np.less else -np.Inf

	def begin(self):
		# Convert names to tensors if given
		graph = tf.get_default_graph()
		self.monitor = graph.as_graph_element(self.monitor)
		if isinstance(self.monitor, tf.Operation):
			self.monitor = self.monitor.outputs[0]

	def before_run(self, run_context):  # pylint: disable=unused-argument
		return session_run_hook.SessionRunArgs(self.monitor)

	def after_run(self, run_context, run_values):
		current = run_values.results

		if self.monitor_op(current - self.min_delta, self.best):
			self.best = current
			self.wait = 0
		else:
			self.wait += 1
			if self.wait >= self.patience:
				run_context.request_stop()


def inference(images):
	"""Build the hw2 model.

	Args:
	  images: Images returned from distorted_inputs() or inputs().

	Returns:
	  Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	#
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 64],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv1)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm1')

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											 shape=[5, 5, 64, 64],
											 stddev=5e-2,
											 wd=None)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv2)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					  name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
						   strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		# images is a batch of input images, so images.get_shape().as_list()[0] is the
		# number of pictures, this op is equivalent to flattent
		reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
		dim = reshape.get_shape()[1].value
		# for weights, num_rows stands for num_features, num_columns stands for num_output_neurons
		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
											  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)

	# linear layer(WX + b),
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
											  stddev=1 / 192.0, wd=None)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
								  tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear
