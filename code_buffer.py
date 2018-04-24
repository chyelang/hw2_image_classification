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