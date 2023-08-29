import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers


writer = tf.summary.create_file_writer('logs/graph_vis')

@tf.function
def my_function(x,y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

tf.summary.trace_on(graph=True, profiler=True)
out = my_function(x, y)

with writer.as_default():
    tf.summary.trace_export(
        name="function_trace",
        step=0,
        profiler_outdir='logs\\graph_vis\\')