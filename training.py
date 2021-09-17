import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
SD = 123
width = 224
height = 224
data_dir = r"dataset"
print("[INFO] loading images...")

training = tf.keras.preprocessing.image_dataset_from_directory(
	data_dir,
	validation_split=0.3,
	subset='training',
	seed=SD,
	image_size=(height, width),
	batch_size=BS
)

validation = tf.keras.preprocessing.image_dataset_from_directory(
	data_dir,
	validation_split=0.3,
	subset='validation',
	seed = SD,
	image_size=(height, width),
	batch_size=BS,
)

classes = training.class_names
classes