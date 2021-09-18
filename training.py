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


for images, labels in training.take(1):
	plt.imshow(images[1].numpy().astype('uint8'))
	plt.title(classes[labels[1]])

model = MobileNetV2(weights='imagenet')
model.compile(optimizer='adam', 
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
metrics=['accuracy'])

model.summary()
face_mask_detection = model.fit(training, validation_data=validation,epochs=3)

img = tf.keras.preprocessing.image.load_img('new2.png', target_size=(height, width))
image_array = tf.keras.preprocessing.image.img_to_array(img)
image_array = tf.expand_dims(image_array, 0)
image_array.shape

predictions = model.predict(image_array)
score = tf.nn.softmax(predictions[0])

print(score)

print("[INFO] saving mask detector model...")
# model.save("MaskDetector.h5")
model.save('MaskDetector.model', save_format="h5")
