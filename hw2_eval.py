"""
Evaluation for hw2.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import configparser

curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('section', "lenovo",
# 						   """where to run this code""")
import hw2

section = FLAGS.section
config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)

tf.app.flags.DEFINE_string('eval_dir', config.get(section, 'eval_dir'),
						   """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', config.get(section, 'eval_data'),
						   """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', config.get(section, 'checkpoint_dir'),
						   """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', config.getint(section, 'eval_interval_secs'),
							"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', config.getint(section, 'num_examples'),
							"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', bool(config.getint(section, 'run_once')),
							"""Whether to run eval only once.""")

def eval_once(saver, summary_writer, top_k_op, summary_op, acc):
	"""Run Eval once.

	Args:
	  saver: Saver.
	  summary_writer: Summary writer.
	  top_k_op: Top K op.
	  summary_op: Summary op.
	"""
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/hw2_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
												 start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0  # Counts the number of correct predictions.
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1

			# Compute precision @ 1.
			precision = true_count / total_sample_count
			print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
			acc_op = acc.assign(precision)
			sess.run(acc_op)

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 1', simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
		return precision


def evaluate():
	"""Eval hw2 for a number of steps."""
	with tf.Graph().as_default() as g:
		# Get images and labels for hw2.
		is_test_eval = FLAGS.eval_data == 'test'
		images, labels = hw2.inputs(is_test_eval=is_test_eval)

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = hw2.inference(images)

		# Calculate predictions.
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
			hw2.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
		acc = tf.Variable(0, dtype=tf.float32)

		while True:
			precision = eval_once(saver, summary_writer, top_k_op, summary_op, acc)
			if FLAGS.run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)
	return precision


def main(argv=None):
	if tf.gfile.Exists(FLAGS.eval_dir):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate()


if __name__ == '__main__':
	tf.app.run()
