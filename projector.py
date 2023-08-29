import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

from tensorboard.plugins.hparams import api as hp

from utils import plot_to_projector

# Load cifar10 from tensorflow_datasets
(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Nomalize function
def normalize_img(image, label):
    return tf.cast(image, tf.float32), label

# Augmentation function
def augment(image, label):
    
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # matplotlib wants [0, 1] values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# Train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)


class_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

x_batch, y_batch = next(iter(ds_train))
plot_to_projector(x_batch, x_batch, y_batch, class_name, log_dir="projector")