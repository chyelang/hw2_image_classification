import tensorflow as tf
import math
import numpy as np

# to use tensorflow-defind convlution function
# from tensorflow.python.ops.gen_nn_ops import conv2d

# to use user-defined convlution function
def conv2d(input, filter, strides, padding = 'SAME'):
	return conv2d_func(input, filter, strides, padding=padding)

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
	# attention to the shape paras!!!!
	W_conv_3x3 = _variable_with_weight_decay('W_conv_3x3',
												shape=[3, 3, reduce1x1_size, map_size[1]],
												stddev=5e-2,
												wd=None)
	b_conv_3x3 = _variable_on_cpu('b_conv_3x3', [map_size[1]], tf.constant_initializer(0.0))

	# follows 1x1_3
	W_conv_5x5 = _variable_with_weight_decay('W_conv_5x5',
												shape=[5, 5, reduce1x1_size, map_size[2]],
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

def inception_v2_module(feed, feed_dim=256, map_size=(128,192,96,64), reduce1x1_size=64, batch_norm=False):
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
												shape=[3, 3, reduce1x1_size, map_size[1]],
												stddev=5e-2,
												wd=None)
	b_conv_3x3 = _variable_on_cpu('b_conv_3x3', [map_size[1]], tf.constant_initializer(0.0))

	# follows 1x1_3
	W_conv_3x3_1 = _variable_with_weight_decay('W_conv_3x3_1',
												shape=[3, 3, reduce1x1_size, map_size[2]],
												stddev=5e-2,
												wd=None)
	b_conv_3x3_1 = _variable_on_cpu('b_conv_3x3_1', [map_size[2]], tf.constant_initializer(0.0))

	# follows 3x3_1
	W_conv_3x3_2 = _variable_with_weight_decay('W_conv_3x3_2',
												shape=[3, 3, map_size[2], map_size[2]],
												stddev=5e-2,
												wd=None)
	b_conv_3x3_2 = _variable_on_cpu('b_conv_3x3_2', [map_size[2]], tf.constant_initializer(0.0))

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
	conv_3x3_1 = conv2d_s1(conv_1x1_3, W_conv_3x3_1) + b_conv_3x3_1
	conv_3x3_2 = conv2d_s1(conv_3x3_1, W_conv_3x3_2) + b_conv_3x3_2
	maxpool1 = max_pool_3x3_s1(feed)
	conv_1x1_4 = conv2d_s1(maxpool1, W_conv_1x1_4) + b_conv_1x1_4

	# concatenate all the feature maps and hit them with a relu
	concat = tf.concat([conv_1x1_1, conv_3x3, conv_3x3_2, conv_1x1_4], 3)
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
		pre_activation = tf.nn.dropout(pre_activation, keep_prob=keep_prob, name="dropout")
	after_activation = tf.nn.relu(pre_activation, name='activated_out')
	_activation_summary(after_activation)

	return after_activation


def spatial_dropout(x, keep_prob, seed=1234, name='dropout'):
	with tf.variable_scope(name):
		# x is a convnet activation with shape BxWxHxF where F is the
		# number of feature maps for that layer
		# keep_prob is the proportion of feature maps we want to keep

		# get the batch size and number of feature maps
		num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

		# get some uniform noise between keep_prob and 1 + keep_prob
		random_tensor = keep_prob
		random_tensor += tf.random_uniform(num_feature_maps,
										   seed=seed,
										   dtype=x.dtype)

		# if we take the floor of this, we get a binary matrix where
		# (1-keep_prob)% of the values are 0 and the rest are 1
		binary_tensor = tf.floor(random_tensor)

		# Reshape to multiply our feature maps by this tensor correctly
		binary_tensor = tf.reshape(binary_tensor,
								   [-1, 1, 1, tf.shape(x)[3]])
		# Zero out feature maps where appropriate; scale up to compensate
		ret = tf.div(x, keep_prob) * binary_tensor
		return ret


def conv2d_func(input, filter, strides, padding='SAME'):
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
	filter_dims = filter.get_shape().as_list()
	filter_mat = tf.reshape(filter, [-1, filter_dims[-1]])
	image_mat = tf.extract_image_patches(images=input, ksizes=[1, filter_dims[0], filter_dims[1], 1], strides=strides,
										 rates=[1, 1, 1, 1], padding=padding)
	image_patches_dim = image_mat.get_shape().as_list()
	image_mat = tf.reshape(image_mat, [-1, image_patches_dim[-1]])
	conv_mat = tf.matmul(filter_mat, image_mat, transpose_a=True, transpose_b=True)
	conv = tf.reshape(tf.transpose(conv_mat), [image_patches_dim[0], image_patches_dim[1], image_patches_dim[2], filter_dims[-1]])
	return conv

def conv2d_test():
	# test case 0
	filter1 = tf.constant([[1,0],[2,1]], dtype=tf.float32)
	filter2 =  tf.constant([[2,3],[1,1]], dtype=tf.float32)
	filter1 = tf.reshape(filter1, [2,2,1,1])
	filter2 = tf.reshape(filter2, [2, 2, 1, 1])
	filter = tf.concat([filter1, filter2], axis=3)
	image = tf.constant([[3,1,2],[1,0,1],[2,1,3]], dtype=tf.float32)
	image = tf.reshape(image, [1,3,3,1])
	actual = conv2d_func(image, filter, [1, 1, 1, 1], padding='VALID')
	expected = tf.nn.conv2d(image, filter, strides=[1, 1, 1, 1], padding='VALID')
	with tf.Session() as sess:
		# print(sess.run(actual))
		# print(sess.run(expected))
		print(sess.run(tf.reduce_sum(expected - actual)))

	# test case 1
	# there will be some minor errors between actual and expected
	image2 = tf.random_uniform([5,50,50,3], minval= 0, maxval=10, dtype=tf.float32)
	filter2 = tf.random_uniform([2,2,3,50], minval= 0, maxval=10, dtype=tf.float32)

	actual2 = conv2d_func(image2, filter2, [1, 1, 1, 1], padding='SAME')
	expected2 = tf.nn.conv2d(image2, filter2, strides=[1, 1, 1, 1], padding='SAME')
	actual2_ = conv2d_func(image2, filter2, [1, 2, 2, 1], padding='VALID')
	expected2_ = tf.nn.conv2d(image2, filter2, strides=[1, 2, 2, 1], padding='VALID')

	with tf.Session() as sess:
		# print(sess.run(actual2_))
		# print(sess.run(expected2_))
		# print(sess.run(expected2 - actual2))
		# print(sess.run(expected2_ - actual2_))
		print(sess.run(tf.reduce_sum(expected2 - actual2)))
		print(sess.run(tf.reduce_sum(expected2_ - actual2_)))

	# test case 2
	image1 = np.arange(10 * 10 * 1).reshape(1, 10, 10, 1)
	image1 = tf.convert_to_tensor(image1.astype(np.float32))
	#sobel_x filter
	filter1 = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
	filter1 = tf.reshape(filter1, [3, 3, 1, 1])

	actual1 = conv2d_func(image1, filter1, [1, 1, 1, 1], padding='SAME')
	expected1 = tf.nn.conv2d(image1, filter1, strides=[1, 1, 1, 1], padding='SAME')
	actual1_ = conv2d_func(image1, filter1, [1, 1, 1, 1], padding='VALID')
	expected1_ = tf.nn.conv2d(image1, filter1, strides=[1, 1, 1, 1], padding='VALID')

	with tf.Session() as sess:
		# print(sess.run(actual1_))
		# print(sess.run(expected1_))
		print(sess.run(tf.reduce_sum(expected1 - actual1)))
		print(sess.run(tf.reduce_sum(expected1_ - actual1_)))

if __name__ == '__main__':
	conv2d_test()
