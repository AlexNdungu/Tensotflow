import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import sklearn.metrics
from tensorboard.plugins import projector
import cv2
import os
import shutil


def plot_to_image(figure):

    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def image_grid(data, labels, class_names):
    
    assert data.ndim == 4

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        plt.subplot(size, size, i + 1, title=class_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        if data.shape[3] == 1:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])

    return figure    


def get_confusion_matrix(y_labels, logits, class_names):
    
    preds = np.argmax(logits, axis=1)
    cm = sklearn.metrics.confusion_matrix(y_labels, preds, labels=range(len(class_names)))

    return cm

def plot_confusion_matrix(cm, class_names):
    
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)

    threshold = cm.max() / 2.0

    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cm_image = plot_to_image(figure)
    return cm_image


def create_sprite(data):

    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant', constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data

def plot_to_projector(x, feature_vector, y, class_names, log_dir='default_log_dir', meta_file='metadata.tsv'):

    assert x.ndim == 4
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir)

    SPRITES_FILE = os.path.join(log_dir, 'sprites.png')
    sprite = create_sprite(x)
    cv2.imwrite(SPRITES_FILE, sprite)

    labels = [class_names[y[i]] for i in range(int(y.shape[0]))]
    with open(os.path.join(log_dir, meta_file), 'w') as f:
        for label in labels:
            f.write(label + '\n')
    
    if feature_vector.ndim != 2:

        print("NOTE: feature_vector should be 2D array. Reshaping...")

        feature_vector = tf.reshape(feature_vector, [feature_vector.shape[0], -1])
        feature_vector = tf.Variable(feature_vector)
        checkpoint = tf.train.Checkpoint(embedding=feature_vector)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = meta_file
        embedding.sprite.image_path = "sprites.png"
        embedding.sprite.single_image_dim.extend([x.shape[1], x.shape[2]])
        projector.visualize_embeddings(log_dir, config)
    