from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import configparser
import re

curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import tensorflow as tf
import csv

# parse arguments passed by command line by FLAGS
FLAGS = tf.app.flags.FLAGS
# Attenten: you need to comment out the following 2 lines in hw2_eval.py to before running hw2_train.py or hw2_train_multi_gpu.py
tf.app.flags.DEFINE_string('section', "lenovo",
						   """where to run this code""")

import hw2
import hw2_eval

section = FLAGS.section
config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)

tf.app.flags.DEFINE_string('log_path', config.get(section, 'log_path'),
						   """Directory where to write event logs """
						   """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', config.getint(section, 'max_steps'),
							"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', bool(config.getint(section, 'log_device_placement')),
							"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', config.getint(section, 'log_frequency'),
							"""How often to log results to the console.""")
tf.app.flags.DEFINE_integer('save_checkpoint_steps', config.getint(section, 'save_checkpoint_steps'),
							"""save_checkpoint_steps""")
tf.app.flags.DEFINE_integer('num_gpus', config.getint(section, 'num_gpus'),
							"""How many GPUs to use.""")


def tower_loss(scope, images, labels):
	"""Calculate the total loss on a single tower running the hw2 model.

	Args:
	  scope: unique prefix string identifying the hw2 tower, e.g. 'tower_0'
	  images: Images. 4D tensor of shape [batch_size, height, width, 3].
	  labels: Labels. 1D tensor of shape [batch_size].

	Returns:
	   Tensor of shape [] containing the total loss for a batch of data
	"""

	# Build inference Graph.
	logits = hw2.inference(images)

	# Build the portion of the Graph calculating the losses. Note that we will
	# assemble the total_loss using a custom function below.
	_ = hw2.loss(logits, labels)

	# Assemble all of the losses for the current tower only.
	losses = tf.get_collection('losses', scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
		# session. This helps the clarity of presentation on tensorboard.
		loss_name = re.sub('%s_[0-9]*/' % hw2.TOWER_NAME, '', l.op.name)
		tf.summary.scalar(loss_name, l)

	top_k_op = tf.nn.in_top_k(logits, labels, 1)
	train_acc = tf.Variable(0, trainable=False, dtype=tf.float32, name="train_acc")
	train_acc_op = tf.assign(train_acc, tf.div(tf.cast(tf.reduce_sum(tf.cast(top_k_op, tf.int32)), tf.float32),
											   tf.cast(FLAGS.batch_size, tf.float32)))
	tf.summary.scalar("train_acc", train_acc_op)

	return total_loss, train_acc_op


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.

	Note that this function provides a synchronization point across all towers.

	Args:
	  tower_grads: List of lists of (gradient, variable) tuples. The outer list
		is over individual gradients. The inner list is over the gradient
		calculation for each tower.
	Returns:
	   List of pairs of (gradient, variable) where the gradient has been averaged
	   across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads


def train():
	"""Train hw2 for a number of steps."""
	csvfile_path = FLAGS.log_path + '/' + time.strftime('%m%d%H%M', time.localtime(time.time()))+'_val_acc.csv'
	with open(csvfile_path, 'a') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t')
		writer.writerow(['global_step', 'train_acc', 'val_acc'])

	with tf.Graph().as_default(), tf.device('/cpu:0'):
		# Create a variable to count the number of train() calls. This equals the
		# number of batches processed * FLAGS.num_gpus.
		global_step = tf.get_variable(
			'global_step', [],
			initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)

		# Calculate the learning rate schedule.
		lr = tf.Variable(0.001, trainable=False, dtype=tf.float32)
		lr_decrease_op = tf.assign(lr, tf.divide(lr, 2.0))

		# Create an optimizer that performs gradient descent.
		opt = tf.train.AdamOptimizer(learning_rate=lr)

		images, labels = hw2.distorted_inputs()
		batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
			[images, labels], capacity=2 * FLAGS.num_gpus)
		# Calculate the gradients for each model tower.
		tower_grads = []
		tower_train_acc_op = []
		keep_prob = []

		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(FLAGS.num_gpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('%s_%d' % (hw2.TOWER_NAME, i)) as scope:
						# Dequeues one batch for the GPU
						image_batch, label_batch = batch_queue.dequeue()
						# Calculate the loss for one tower of the CIFAR model. This function
						# constructs the entire CIFAR model but shares the variables across
						# all towers.
						loss, train_acc_op = tower_loss(scope, image_batch, label_batch)

						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()

						# Retain the summaries from the final tower.
						summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

						# Calculate the gradients for the batch of data on this CIFAR tower.
						grads = opt.compute_gradients(loss)

						# Keep track of the gradients across all towers.
						tower_grads.append(grads)
						tower_train_acc_op.append(train_acc_op)
						keep_prob.append(tf.get_default_graph().get_tensor_by_name(scope + 'keep_prob2:0'))
						keep_prob.append(tf.get_default_graph().get_tensor_by_name(scope + 'keep_prob3:0'))
						keep_prob.append(tf.get_default_graph().get_tensor_by_name(scope + 'dense1/keep_prob:0'))

		# We must calculate the mean of each gradient. Note that this is the
		# synchronization point across all towers.
		grads = average_gradients(tower_grads)
		train_acc_op_avg = tf.divide(tf.reduce_sum(tower_train_acc_op), FLAGS.num_gpus)

		# Add a summary to track the learning rate.
		summaries.append(tf.summary.scalar('learning_rate', lr))
		summaries.append(tf.summary.scalar('train_acc_avg', train_acc_op_avg))

		# Add histograms for gradients.
		for grad, var in grads:
			if grad is not None:
				summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

		# Apply the gradients to adjust the shared variables.
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram(var.op.name, var))

		# Track the moving averages of all trainable variables.
		variable_averages = tf.train.ExponentialMovingAverage(
			hw2.MOVING_AVERAGE_DECAY, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		# Group all updates to into a single train op.
		train_op = tf.group(apply_gradient_op, variables_averages_op)


		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = 0
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs([loss, train_acc_op_avg])

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results[0]
					train_acc = run_values.results[1]
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size * FLAGS.num_gpus / duration
					sec_per_batch = float(duration / (FLAGS.log_frequency * FLAGS.num_gpus))
					format_str = ('%s: equal global step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
								  'sec/batch)')
					print(format_str % (datetime.now(), self._step * FLAGS.num_gpus, loss_value, train_acc,
										examples_per_sec, sec_per_batch))


		class _EarlyStoppingHook(tf.train.SessionRunHook):
			"""Hook that requests stop at a specified step."""

			def __init__(self, min_delta=0.01, patience=10):
				self.patience = patience
				self.min_delta = min_delta
				self._ckpt_step = -1
				self.best = -1
				self.wait = 0
				self.current = 0

			def before_run(self, run_context):
				return tf.train.SessionRunArgs([train_acc_op_avg])

			def after_run(self, run_context, run_values):
				ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
				if ckpt and ckpt.model_checkpoint_path:
					cur_ckpt_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
					if cur_ckpt_step > self._ckpt_step:
						self._ckpt_step = cur_ckpt_step
						self.current = hw2_eval.evaluate(1)
						format_str = '%s: step %d, val_acc = %.3f'
						print(format_str % (datetime.now(), self._ckpt_step, self.current))

						with open(csvfile_path, 'a') as csvfile:
							writer = csv.writer(csvfile, delimiter='\t')
							writer.writerow([self._ckpt_step * 2, run_values.results[0], self.current])

						if (self.current - self.min_delta) > self.best:
							self.best = self.current
							self.wait = 0
						else:
							self.wait += 1
							if self.wait >= self.patience / 2:
								print('Divide lr by 2!')
								run_context.session.run(lr_decrease_op)
							if self.wait >= self.patience:
								print('Early stop training!')
								print('val_acc log stored in {0}'.format(csvfile_path))
								run_context.request_stop()


		config_tf = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True)

		feed_dict = {}
		for counter, item in enumerate(keep_prob, 1):
			if counter % 3 == 1:
				feed_dict[item] = 0.75
			elif counter % 3 == 2:
					feed_dict[item] = 0.75
			else:
				feed_dict[item] = 0.5
		early_stop_hook = _EarlyStoppingHook(min_delta=0.00001, patience=10)
		saver = tf.train.Saver(max_to_keep=10)
		ckpt_hook = tf.train.CheckpointSaverHook(
			checkpoint_dir=FLAGS.log_path,
			saver=saver,
			save_steps=FLAGS.save_checkpoint_steps / 2)
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=FLAGS.log_path,
				hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
					   tf.train.NanTensorHook(loss),
					   _LoggerHook(),
					    early_stop_hook,
					   ckpt_hook],
				save_checkpoint_secs=-1,
				log_step_count_steps=100,
				config=config_tf) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op, feed_dict=feed_dict)

def main(argv=None):
	# # why to delete? 因为此处的train_dir只是存放log和checkpoint的，并不是训练数据
	if tf.gfile.Exists(FLAGS.log_path):
		tf.gfile.DeleteRecursively(FLAGS.log_path)
	tf.gfile.MakeDirs(FLAGS.log_path)
	train()


if __name__ == '__main__':
	tf.app.run()
