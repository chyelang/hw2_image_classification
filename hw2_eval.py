"""
Evaluation for hw2. 
*****
commands to run to start your test:
for data_set1 evaluation: 
python hw2_eval.py --eval_dir ./eval_dir --checkpoint_dir ./saved_model/train_log_dset1_handed \
--test_data_path /your/path/to/test_set --num_examples examples_to_run_per_test --top_k 5 --run_once 0

for data_set1 evaluation: 
python hw2_eval.py --eval_dir ./eval_dir --checkpoint_dir ./saved_model/train_log_dset2_handed \
--test_data_path /your/path/to/test_set --num_examples examples_to_run_per_test --top_k 5 --run_once 0

=====
you need to specify 4 parameters: --test_data_path /your/path/to/test_set --num_examples examples_to_run_per_test --top_k 5 --run_once 0
=====

The hw2_eval.py will fetch randomly --num_examples images in --test_data_path for a single test and return the --top_k error, this procedure will repeat if 
--run_once is set 0 (and don't repeat if it's set to 1). 

=====
Attention: it's recommended to set --num_examples as big as your test set to get a consistant test result!
=====

for example:
python hw2_eval.py --section ecm --eval_dir ./eval_dir --checkpoint_dir ./saved_model/train_log_dset1_handed \
--test_data_path /scratch/xzou/hw2_image_classification/modified_data/dset1/test --num_examples 900 --top_k 5 --run_once 0

python hw2_eval.py --section ecm --eval_dir ./eval_dir --checkpoint_dir ./saved_model/train_log_dset2_handed \
--test_data_path /scratch/xzou/hw2_image_classification/modified_data/dset2/test --num_examples 900 --top_k 5 --run_once 0

*****
validation accurary (be done for about 800 samples in validation set):
for dset1: top1 acc ~ 0.52, top5 acc ~ 0.79
for dset2: top1 acc ~ 0.63, top5 acc ~ 0.84
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
# you need to comment out the following 2 lines to before running hw2_train.py or hw2_train_multi_gpu.py
tf.app.flags.DEFINE_string('section', "lenovo",
						   """where to run this code""")
import hw2

section = FLAGS.section
config = configparser.RawConfigParser()
config_path = projectRootPath + '/' + 'config.cfg'
config.read(config_path)
is_during_train = True

tf.app.flags.DEFINE_string('eval_dir', config.get(section, 'eval_dir'),
						   """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', config.get(section, 'checkpoint_dir'),
						   """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_string('eval_data', config.get(section, 'eval_data'),
						   """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', config.getint(section, 'eval_interval_secs'),
							"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', config.getint(section, 'num_examples'),
							"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', bool(config.getint(section, 'run_once')),
							"""Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('top_k', config.getint(section, 'k'),
							"""k for top_k accuracy""")

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
		if ckpt and is_during_train and ckpt.model_checkpoint_path:
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		# use best checkpoint for eval after the trainning has finished
		elif ckpt and (not is_during_train) and ckpt.all_model_checkpoint_paths[0]:
			# Restores from checkpoint
			# saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
			# global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

			# specially for TA's test
			if "dset1" in FLAGS.checkpoint_dir:
				ckpt_path = FLAGS.checkpoint_dir + "/model.ckpt-4051"
			else:
				ckpt_path = FLAGS.checkpoint_dir + "/model.ckpt-5101"
			saver.restore(sess, ckpt_path)
			global_step = ckpt_path.split('/')[-1].split('-')[-1]
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
			print('%s: precision @ Top%d = %.3f' % (datetime.now(), FLAGS.top_k, precision))
			acc_op = acc.assign(precision)
			sess.run(acc_op)

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ Top%d'%(FLAGS.top_k), simple_value=precision)
			summary_writer.add_summary(summary, global_step)
		except Exception as e:
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
		return precision


def evaluate(k):
	"""Eval hw2 for a number of steps."""
	with tf.Graph().as_default() as g:
		# Get images and labels for hw2.
		is_test_eval = FLAGS.eval_data == 'test'
		images, labels = hw2.inputs(is_test_eval=is_test_eval)

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = hw2.inference(images)

		# Calculate predictions.
		top_k_op = tf.nn.in_top_k(logits, labels, k)

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
	evaluate(FLAGS.top_k)


if __name__ == '__main__':
	is_during_train = False
	tf.app.run()
