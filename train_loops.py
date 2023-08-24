import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

# Load mnist from tensorflow_datasets
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Define preprocessing function
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# Train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# Create model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ],
    name="model",
)

num_epoch = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accu_metric = keras.metrics.SparseCategoricalAccuracy()

# Training loop
for epoch in range(num_epoch):
    print("\nStart of epoch %d" % (epoch,))
    for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        accu_metric.update_state(y_batch, y_pred)
    
    train_acc = accu_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    accu_metric.reset_states()

# Test loop
for batch_idx, (x_batch, y_batch) in enumerate(ds_test):
    y_pred = model(x_batch, training=True)
    accu_metric.update_state(y_batch, y_pred)


train_acc = accu_metric.result()
print("Training acc over test set: %.4f" % (float(train_acc),))
accu_metric.reset_states()