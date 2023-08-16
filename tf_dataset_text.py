import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

#Tokenizer

tokenizer =  tfds.deprecated.text.Tokenizer()

def build_vocabulary():

    vocabulaty = set()

    for text, _ in ds_train:

        vocabulaty.update(tokenizer.tokenize(text.numpy().lower()))

    return vocabulaty

vocabulary = build_vocabulary()

encoder = tfds.deprecated.text.TokenTextEncoder(
    vocabulary, oov_token="<UNK>", lowercase = True, tokenizer=tokenizer
)

def my_encoding(text_tensor, label):

    return encoder.encode(text_tensor.numpy()), label

def encode_map(text, label):

    encoded_text, label = tf.py_function(
        my_encoding, inp=[text, label], Tout=(tf.int64, tf.int64)
    )

    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train.map(encode_map, num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32, padded_shapes = ([None], ()))
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(encode_map)
ds_test = ds_test.padded_batch(32, padded_shapes = ([None], ()))

model = keras.Sequential([
    keras.layers.Masking(mask_value=0),
    keras.layers.Embedding(input_dim=len(vocabulary)+2, output_dim=32),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1),
])

model.compile(
    optimizer=keras.optimizers.Adam(3e-1, clipnorm=1),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=10, verbose=2)

model.evaluate(ds_test)