import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot


(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

print(ds_info)