from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import tensorflow as tf
import configparser

import hw2_input
import layers
from utils import _variable_on_cpu
from utils import _variable_with_weight_decay
from utils import _activation_summary

# parse arguments passed by command line by FLAGS
FLAGS = tf.app.flags.FLAGS

section = FLAGS.section
config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = config.get(section, 'CUDA_VISIBLE_DEVICES')

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', config.getint(section, 'batch_size'),
							"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('train_data_path', config.get(section, 'train_data_path'),
						   """Path to the hw2 train data directory.""")
tf.app.flags.DEFINE_string('test_data_path', config.get(section, 'test_data_path'),
						   """Path to the hw2 test data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', bool(config.getint(section, 'use_fp16')),
							"""Train the model using fp16.""")

# Global constants describing the hw2 data set.
IMAGE_SIZE = hw2_input.IMAGE_SIZE
NUM_CLASSES = hw2_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = hw2_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = hw2_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 70  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.2  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate. for gradient desent
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate. for adam


def distorted_inputs():
	"""Construct distorted input for CIFAR training using the Reader ops.

	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.

	Raises:
	  ValueError: If no data_dir
	"""
	if not FLAGS.train_data_path:
		raise ValueError('Please supply a data_dir')
	images, labels = hw2_input.distorted_inputs(data_dir=FLAGS.train_data_path,
												batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels


def inputs(is_test_eval):
	"""Construct input for hw2 evaluation using the Reader ops.

	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.

	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.

	Raises:
	  ValueError: If no data_dir
	"""
	if is_test_eval:
		if not FLAGS.test_data_path:
			raise ValueError('Please supply test data path')
		images, labels = hw2_input.inputs(is_test_eval=is_test_eval,
										  data_dir=FLAGS.test_data_path,
										  batch_size=FLAGS.batch_size)
	else:
		if not FLAGS.train_data_path:
			raise ValueError('Please supply train data path')
		images, labels = hw2_input.inputs(is_test_eval=is_test_eval,
										  data_dir=FLAGS.train_data_path,
										  batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
	return images, labels

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
	# conv_stack1
	with tf.variable_scope('conv_stack1') as scope:
		kernel_list = [[3,3,3,64], [3,3,64,64]]
		stride_list = [[1,1,1,1], [1,2,2,1]]
		padding_list = ['SAME', 'SAME']
		conv_stack1 = layers.conv2d_stack(images, kernel_list, stride_list, padding_list, batch_norm = True)

	# pool1
	pool1 = tf.nn.max_pool(conv_stack1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
						   padding='SAME', name='pool1')

	with tf.variable_scope('conv_stack2') as scope:
		kernel_list = [[3,3,64,128],[3,3,128,128]]
		stride_list = [[1,2,2,1],[1,1,1,1]]
		padding_list = ['SAME','SAME']
		conv_stack2 = layers.conv2d_stack(pool1, kernel_list, stride_list, padding_list, batch_norm = True)

	# pool2
	pool2 = tf.nn.max_pool(conv_stack2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool2')

	with tf.variable_scope('conv_stack3') as scope:
		kernel_list = [[3,3,128,256],[3,3,256,256]]
		stride_list = [[1,1,1,1],[1,1,1,1]]
		padding_list = ['SAME','SAME']
		conv_stack3 = layers.conv2d_stack(pool2, kernel_list, stride_list, padding_list, batch_norm = True)

	# pool3
	pool3 = tf.nn.max_pool(conv_stack3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool3')

	# inception1
	with tf.variable_scope('inception1') as scope:
		inception1 = layers.inception_v1_module(pool3, 256, map_size=(128, 192, 96, 64), reduce1x1_size=64, batch_norm=True)

	# pool4
	pool4 = tf.nn.max_pool(inception1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
						   padding='SAME', name='pool4')

	# inception2
	with tf.variable_scope('inception2') as scope:
		inception2 = layers.inception_v1_module(pool4, 480, map_size=(128, 192, 96, 64), reduce1x1_size=64, batch_norm=True)

	# pool5
	pool5 = tf.nn.max_pool(inception2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						   padding='SAME', name='pool5')

	# dense1
	with tf.variable_scope('dense1') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		# images is a batch of input images, so images.get_shape().as_list()[0] is the
		# number of pictures, this op is equivalent to flattent
		reshape = tf.reshape(pool5, [images.get_shape().as_list()[0], -1])
		dim = reshape.get_shape()[1].value
		keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
		dense1 = layers.dense_layer(reshape, dim, 2048, keep_prob=keep_prob, batch_norm=False, weight_decay=1e-4)
		tf.summary.scalar("keep_prob",keep_prob)

	# dense2
	with tf.variable_scope('dense2') as scope:
		keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
		dense2 = layers.dense_layer(dense1, 2048, 1024, keep_prob=keep_prob, batch_norm=False, weight_decay=1e-4)
		tf.summary.scalar("keep_prob", keep_prob)

	# linear layer(WX + b),
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
											  stddev=1 / 1024.0, wd=None)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
								  tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(dense2, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear

def loss(logits, labels):
	"""Add L2Loss to all the trainable variables.

	Add summary for "Loss" and "Loss/avg".
	Args:
	  logits: Logits from inference().
	  labels: Labels from distorted_inputs or inputs(). 1-D tensor
			  of shape [batch_size]

	Returns:
	  Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
	"""Add summaries for losses in hw2 model.

	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
	  total_loss: Total loss from loss().
	Returns:
	  loss_averages_op: op for generating moving averages of losses.
	"""
	# Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	## losses + [total_loss] 结果是一个list，包含两个元素，即individual losses and the total loss
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):
	"""Train hw2 model.

	Create an optimizer and apply to all trainable variables. Add moving
	average for all trainable variables.

	Args:
	  total_loss: Total loss from loss().
	  global_step: Integer Variable counting the number of training steps
		processed.
	Returns:
	  train_op: op for training.
	"""
	# Variables that affect learning rate.
	# # NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 扫过多少example之后才算一个epoch，一般是设置成样本总数吧？
	# # FLAGS.batch_size = 每个batch选取多少个example
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	# # 扫描完一个batch之后，梯度更新了一次，算一个step，算运行了一次graph。 下式计算出需要扫描多少个batch之后才进行一次lr decay
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the number of steps.
	##　decayed_learning_rate = learning_rate *　decay_rate ^ (global_step / decay_steps)
	## 注意由于staircase=True， 所以指数一直是一个int!
	# lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
	#                                 global_step,
	#                                 decay_steps,
	#                                 LEARNING_RATE_DECAY_FACTOR,
	#                                 staircase=True)
	# tf.summary.scalar('learning_rate', lr)

	# Generate moving averages of all losses and associated summaries.
	loss_averages_op = _add_loss_summaries(total_loss)
	global_step = tf.train.get_or_create_global_step()
	lr = tf.cond(tf.less(global_step, 10000),
				 lambda: tf.constant(0.005),
				 lambda: tf.cond(tf.less(global_step, 20000),
								 lambda: tf.constant(0.001),
								 lambda: tf.cond(tf.less(global_step, 30000),
												 lambda: tf.constant(0.0005),
												 lambda: tf.constant(0.00025))))
	tf.summary.scalar('learning_rate', lr)
	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		# opt = tf.train.GradientDescentOptimizer(lr)
		opt = tf.train.AdamOptimizer(learning_rate=lr)
		lr_actu = opt._lr
		tf.summary.scalar('learning_rate_actu', lr_actu)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	with tf.control_dependencies([apply_gradient_op]):
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

	return variables_averages_op
