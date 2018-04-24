from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import configparser

curPath = os.path.abspath(os.path.dirname(__file__))
projectRootPath = curPath
import tensorflow as tf

# parse arguments passed by command line by FLAGS
FLAGS = tf.app.flags.FLAGS
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


def train():
	"""Train hw2 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()

		# Get images and labels for hw2.
		# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
		# GPU and resulting in a slow down.
		with tf.device('/cpu:0'):
			images, labels = hw2.distorted_inputs()

		# Build a Graph that computes the logits predictions from the
		# inference model.
		logits = hw2.inference(images)

		# Calculate loss.
		loss = hw2.loss(logits, labels)

		# Build a Graph that trains the model with one batch of examples and
		# updates the model parameters.
		train_op = hw2.train(loss, global_step)

		class _LoggerHook(tf.train.SessionRunHook):
			"""Logs loss and runtime."""

			def begin(self):
				self._step = -1
				self._start_time = time.time()

			def before_run(self, run_context):
				self._step += 1
				return tf.train.SessionRunArgs(loss)  # Asks for loss value.

			def after_run(self, run_context, run_values):
				if self._step % FLAGS.log_frequency == 0:
					current_time = time.time()
					duration = current_time - self._start_time
					self._start_time = current_time

					loss_value = run_values.results
					examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
					sec_per_batch = float(duration / FLAGS.log_frequency)

					format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
								  'sec/batch)')
					print(format_str % (datetime.now(), self._step, loss_value,
										examples_per_sec, sec_per_batch))

		class _EarlyStoppingHook(tf.train.SessionRunHook):
			"""Hook that requests stop at a specified step."""

			def __init__(self, min_delta=0.001, patience=10):
				self.patience = patience
				self.min_delta = min_delta
				self._ckpt_step = -1
				self.best = -1
				self.wait = 0

			def after_run(self, run_context, run_values):
				ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
				if ckpt and ckpt.model_checkpoint_path:
					cur_ckpt_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
					if cur_ckpt_step > self._ckpt_step:
						self._ckpt_step = cur_ckpt_step
						current = hw2_eval.evaluate()
						format_str = '%s: step %d, val_acc = %.2f'
						print(format_str % (datetime.now(), self._ckpt_step, current))
						if (current - self.min_delta) > self.best:
							self.best = current
							self.wait = 0
						else:
							self.wait += 1
							if self.wait >= self.patience:
								print('Early stop training!')
								run_context.request_stop()

		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=FLAGS.log_path,
				hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
					   tf.train.NanTensorHook(loss),
					   _LoggerHook(),
					   _EarlyStoppingHook(min_delta=0.001, patience=10)],
				config=tf.ConfigProto(
					log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while not mon_sess.should_stop():
				mon_sess.run(train_op)


def main(argv=None):
	# # why to delete? 因为此处的train_dir只是存放log和checkpoint的，并不是训练数据
	if tf.gfile.Exists(FLAGS.log_path):
		tf.gfile.DeleteRecursively(FLAGS.log_path)
	tf.gfile.MakeDirs(FLAGS.log_path)
	train()

if __name__ == '__main__':
	tf.app.run()
