import re
import sys
import math
import datetime
import tensorflow as tf
import configparser
import numpy as np
import logging
FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'

from tensorflow.python.ops.gen_nn_ops import conv2d
# from hw2 import _variable_on_cpu
# from hw2 import _variable_with_weight_decay
# from hw2 import _activation_summary


def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
	  x: Tensor
	Returns:
	  nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity',
					  tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensor
	"""
	# ('/cpu:0')########### /device:GPU:0
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


## wd is lambda in regularization
def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
	  name: name of the variable
	  shape: list of ints
	  stddev: standard deviation of a truncated Gaussian
	  wd: add L2Loss weight decay multiplied by this float. If None, weight
		  decay is not added for this Variable.

	Returns:
	  Variable Tensor
	"""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv2d_stack(feed, kernel_list, stride_list, padding_list):
	if not ((len(kernel_list) == len(stride_list)) and (len(stride_list) == len(padding_list))):
		return
	inputs=[]
	inputs.append(feed)
	for i in range(len(kernel_list)):
		with tf.variable_scope('conv%d' %(i+1)) as scope:
			kernel = _variable_with_weight_decay('weights',
												 shape=kernel_list[i],
												 stddev=5e-2,
												 wd=None)
			conv = conv2d(inputs[-1], kernel, stride_list[i], padding=padding_list[i])
			biases = _variable_on_cpu('biases', kernel_list[i][-1], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name='activated_out')
			_activation_summary(conv1)
			inputs.append(conv1)
	return inputs[-1]

def inception_v1_moduel(feed, feed_dim=256, map_size=(128,192,96,64), reduce1x1_size=64, name="inception_v1"):
	"""
	:param feed: 
	:param map_size: lists of number of feature maps output by each tower (1x1, 3x3, 5x5, 1x1) inside the Inception module
	:param reduce1x1_size: number of feature maps output by each 1×1 convolution that precedes a large convolution
	:return: 
	"""

	def conv2d_s1(x, W):
		return conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_3x3_s1(x):
		return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

	# follows input
	W_conv_1x1_1 = _variable_with_weight_decay('W_conv_1x1_1',
												shape=[1, 1, feed_dim, map_size[0]],
												stddev=5e-2,
												wd=None)
	b_conv_1x1_1 = _variable_on_cpu('b_conv_1x1_1', [map_size[0]], tf.constant_initializer(0.0))

	# follows input
	W_conv_1x1_2 = _variable_with_weight_decay('W_conv_1x1_2',
												shape=[1, 1, feed_dim, reduce1x1_size],
												stddev=5e-2,
												wd=None)
	b_conv_1x1_2 = _variable_on_cpu('b_conv_1x1_2', [reduce1x1_size], tf.constant_initializer(0.0))

	# follows input
	W_conv_1x1_3 = _variable_with_weight_decay('W_conv_1x1_3',
												shape=[1, 1, feed_dim, reduce1x1_size],
												stddev=5e-2,
												wd=None)
	b_conv_1x1_3 = _variable_on_cpu('b_conv_1x1_3', [reduce1x1_size], tf.constant_initializer(0.0))

	# follows 1x1_2
	W_conv_3x3 = _variable_with_weight_decay('W_conv_3x3',
												shape=[1, 1, reduce1x1_size, map_size[1]],
												stddev=5e-2,
												wd=None)
	b_conv_3x3 = _variable_on_cpu('b_conv_3x3', [map_size[1]], tf.constant_initializer(0.0))

	# follows 1x1_3
	W_conv_5x5 = _variable_with_weight_decay('W_conv_5x5',
												shape=[1, 1, reduce1x1_size, map_size[2]],
												stddev=5e-2,
												wd=None)
	b_conv_5x5 = _variable_on_cpu('b_conv_5x5', [map_size[2]], tf.constant_initializer(0.0))

	# follows max pooling
	W_conv_1x1_4 = _variable_with_weight_decay('W_conv_1x1_4',
												shape=[1, 1, feed_dim, map_size[3]],
												stddev=5e-2,
												wd=None)
	b_conv_1x1_4 = _variable_on_cpu('b_conv_1x1_4', [map_size[3]], tf.constant_initializer(0.0))

	# Inception Module
	conv_1x1_1 = conv2d_s1(feed, W_conv_1x1_1) + b_conv_1x1_1
	conv_1x1_2 = tf.nn.relu(conv2d_s1(feed, W_conv_1x1_2) + b_conv_1x1_2)
	conv_1x1_3 = tf.nn.relu(conv2d_s1(feed, W_conv_1x1_3) + b_conv_1x1_3)
	conv_3x3 = conv2d_s1(conv_1x1_2, W_conv_3x3) + b_conv_3x3
	conv_5x5 = conv2d_s1(conv_1x1_3, W_conv_5x5) + b_conv_5x5
	maxpool1 = max_pool_3x3_s1(feed)
	conv_1x1_4 = conv2d_s1(maxpool1, W_conv_1x1_4) + b_conv_1x1_4

	# concatenate all the feature maps and hit them with a relu
	concat = tf.concat([conv_1x1_1, conv_3x3, conv_5x5, conv_1x1_4], 3)
	inception = tf.nn.relu(concat, name=name)
	_activation_summary(inception)
	return inception

if __name__ == '__main__':
	print("hello")
