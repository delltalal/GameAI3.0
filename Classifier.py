import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import subprocess
import os
import random as rn

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from os import listdir
from PIL import Image
from keras.layers.convolutional import MaxPooling2D, Conv2D

data_dir = pathlib.Path('datasets/GameAI')

batch_size = 128
img_height = 512
img_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]

)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'), layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'), layers.Dropout(0.5),
    layers.Dense(256, activation='relu'), layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 150

checkpoint_filepath = 'states/'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq='epoch')

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[model_checkpoint_callback]
)

model.load_weights(checkpoint_filepath)

model.save('models/classifier.h5')

