import tensorflow as tf
# parse arguments passed by command line by FLAGS
FLAGS = tf.app.flags.FLAGS
import re
TOWER_NAME = 'tower'

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
	  x: Tensor
	Returns:
	  nothing
	"""
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	# tensor_name =  x.op.name
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
	  name: name of the variable
	  shape: list of ints
	  initializer: initializer for Variable

	Returns:
	  Variable Tensor
	"""
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
		tf.keras.initializers.he_normal(seed=None))
	# tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var