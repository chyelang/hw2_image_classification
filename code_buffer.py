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