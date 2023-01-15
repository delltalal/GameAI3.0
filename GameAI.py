import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import subprocess
import os
import shutil
import torch

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from os import listdir
from PIL import Image

img_height = 512
img_width = 512

class_names = ['Blood', 'Brick', 'Cartoon', 'Concrete', 'Concrete_Painted', 'Covers', 'Cracks', 'Decorative', 'Dirt',
               'Door', 'Fabric', 'Faces', 'Fingerprints', 'Graffiti', 'Grates', 'Ground', 'Ground_Grass', 'Hair',
               'Icons', 'Leaves', 'Metal', 'Metal_Diamond-Metal', 'Plaster', 'Plaster_Damaged-Plaster', 'Rock', 'Rust',
               'Rust_Rusted-Paint', 'Sand', 'Signs', 'Sprites', 'Stone', 'Stone_Stone-Walls', 'Text', 'VFX', 'Wood',
               'Wood_Bark', 'Wood_FibreBoard', 'Wood_Painted', 'Wood_Planks', 'Wood_Shutters']

model = keras.models.load_model('models/classifier.h5')

input_dir = "input"
for images in os.listdir(input_dir):

    # check if the image ends with png
    if (images.endswith(".png") or images.endswith(".jpg") \
            or images.endswith(".jpeg") or images.endswith(".dds") or images.endswith(".tga") \
            or images.endswith(".PNG") or images.endswith(".JPG") or images.endswith(".JPEG") \
            or images.endswith(".DDS") or images.endswith(".TGA")):

        texture_path = pathlib.Path('input/'+images)
        img = tf.keras.utils.load_img(
            texture_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))

        )
        category = class_names[np.argmax(score)]
        try:
            os.mkdir('processing/' + category)
            shutil.copy(texture_path, 'processing/' + category) # Copies images to processing folder


            images = torch(3, img_height, img_width) # channels, height, width
            kernel_size, stride = 512, 512
            patches = images.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
            patches = patches.contiguous().view(patches.size(0), -1, kernel_size, kernel_size)
            print(patches.shape) # channels, patches, kernel_size, kernel_size

            subprocess.call('python output/merge.py')
            subprocess.call('split-image' + ' processing/' + category + '/' + images) # Creates tiles
            subprocess.call('python inference_realesrgan.py -n ' + category + ' -i processing/' + category + ' -o output') # Upscales based on category
            subprocess.call('split-image' + images + ' -r') # Merges tiles
            #shutil.rmtree('processing/'+category) # Deletes the processing folder
        except OSError as error:
            shutil.copy(texture_path, 'processing/' + category)
            #subprocess.call('python inference_realesrgan.py -n ' + category + ' -i processing/' + category + ' -o output')
            #shutil.rmtree('processing/'+category)