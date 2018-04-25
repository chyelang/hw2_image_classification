import tensorflow as tf
import math

from tensorflow.python.ops.gen_nn_ops import conv2d
from utils import _variable_on_cpu
from utils import _variable_with_weight_decay
from utils import _activation_summary

def conv2d_stack(feed, kernel_list, stride_list, padding_list, batch_norm = False):
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
			if batch_norm:
				mean, variance = tf.nn.moments(conv, axes=[0])
				epsilon = 1e-5
				gamma = _variable_on_cpu('gammas', kernel_list[i][-1], tf.constant_initializer(1.0))
				pre_activation = tf.nn.batch_normalization(conv, mean, variance, biases, gamma, epsilon)
			else:
				pre_activation = tf.nn.bias_add(conv, biases)
			after_activation = tf.nn.relu(pre_activation, name='activated_out')
			_activation_summary(after_activation)
			inputs.append(after_activation)
	return inputs[-1]

def inception_v1_module(feed, feed_dim=256, map_size=(128,192,96,64), reduce1x1_size=64, batch_norm=False):
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
	if batch_norm:
		biases = _variable_on_cpu('biases', sum(map_size), tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(concat, axes=[0])
		epsilon = 1e-5
		gamma = _variable_on_cpu('gammas', sum(map_size), tf.constant_initializer(1.0))
		pre_activation = tf.nn.batch_normalization(concat, mean, variance, biases, gamma, epsilon)
	else:
		pre_activation = concat
	after_activation = tf.nn.relu(pre_activation, name='activated_out')
	_activation_summary(after_activation)

	return after_activation


def dense_layer(feed, input_dim, output_dim, dropout=False, keep_prob=None, batch_norm=False, weight_decay=None):
	weights = _variable_with_weight_decay('weights', shape=[input_dim, output_dim],
										  stddev=0.04, wd=weight_decay)
	biases = _variable_on_cpu('biases', [output_dim], tf.constant_initializer(0.1))
	intermediate = tf.matmul(feed, weights)
	if batch_norm:
		mean, variance = tf.nn.moments(intermediate, axes=[0])
		epsilon = 1e-5
		gamma = _variable_on_cpu('gammas', [output_dim], tf.constant_initializer(1.0))
		pre_activation = tf.nn.batch_normalization(intermediate, mean, variance, biases, gamma, epsilon)
	else:
		pre_activation = intermediate + biases
	if dropout:
		pre_activation = tf.nn.dropout(pre_activation, keep_prob=keep_prob, name=dropout)
	after_activation = tf.nn.relu(pre_activation, name='activated_out')
	_activation_summary(after_activation)

	return after_activation

def conv2d_sub(input, filter, strides, padding = 'SAME'):
	# NHWC
	"""

	:param input:
	:param filter:
	:param strides:
	:param padding:
	"VALID" only ever drops the right-most columns (or bottom-most rows).
	"SAME" tries to pad evenly left and right, but if the amount of columns to
	 be added is odd, it will add the extra column to the right, as is the case
	 in this example (the same logic applies vertically: there may be an extra
	 row of zeros at the bottom).
	:return:
	"""
	in_dims = input.get_shape().as_list()
	filter_dims = filter.get_shape().as_list()
	out_dims = [in_dims[0], 0, 0, filter_dims[-1]]
	if padding == "SAME":
		out_dims[1] = math.ceil(float(in_dims[1])/float(strides[1]))
		out_dims[2] = math.ceil(float(in_dims[2]) / float(strides[2]))
		delta_H = strides[1]*(out_dims[1]-1) + filter_dims[1] - in_dims[1]
		delta_W = strides[2] * (out_dims[2] - 1) + filter_dims[2] - in_dims[2]
		paddings = tf.constant([[0, 0], [math.floor(delta_H/2.0), math.ceil(delta_H/2.0)],
								[math.floor(delta_W/2.0), math.ceil(delta_W/2.0)], [0, 0]])
		input = tf.pad(input, paddings, "CONSTANT")

	elif padding == "VALID":
		out_dims[1] = math.ceil(float(in_dims[1] - filter_dims[1] + 1)
								/ float(strides[1]))
		out_dims[2] = math.ceil(float(in_dims[2] - filter_dims[2] + 1)
								/ float(strides[2]))

	else:
		return


if __name__ == '__main__':
	print("hello")
