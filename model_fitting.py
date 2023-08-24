import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist



# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, 3, padding="same"),
        layers.ReLU(),
        layers.Conv2D(128, 3, padding="same"),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="model",
)

# Create a class called CustomFit
class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)    

        return {m.name: m.result() for m in self.metrics}


training = CustomFit(model)
training.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

training.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)