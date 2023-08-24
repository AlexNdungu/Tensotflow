import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Load mnist dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Normalizing the data
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Create the sequential model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

# create a scheduler function
def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

save_callback = keras.callbacks.ModelCheckpoint(
    "checkpoint/", save_weights_only=True, monitor="accuracy", save_best_only=False
)

model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2, callbacks=[save_callback, lr_scheduler])