import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

from tensorboard.plugins.hparams import api as hp

from utils import plot_to_image, image_grid

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
    return tf.cast(image, tf.float32) / 255.0, label

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

def train_model_one_epoch(hparams):

    units = hparams[HP_NUM_UNITS]
    drop_rate = hparams[HP_DROPOUT]
    learning_rate = hparams[HP_LEARNING_RATE]  
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model = keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(units, activation="relu"),
            layers.Dropout(drop_rate),
            layers.Dense(10),
        ]
    )

    for batch_idx, (x, y) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y, y_pred)
    
    # write to tensorboard
    run_dir =(
        "logs/hparam_tuning/" + str(units) + "_" + str(drop_rate) + "_" + str(learning_rate)
    )

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy = acc_metric.result()
        tf.summary.scalar('accuracy', accuracy, step=1)

    acc_metric.reset_states()



loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64, 128]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.3, 0.5]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4, 1e-5]))


for lr in HP_LEARNING_RATE.domain.values:
    for units in HP_NUM_UNITS.domain.values:
        for drop_rate in HP_DROPOUT.domain.values:
            hparams = {
                HP_NUM_UNITS: units,
                HP_DROPOUT: drop_rate,
                HP_LEARNING_RATE: lr,
            }
            train_model_one_epoch(hparams)