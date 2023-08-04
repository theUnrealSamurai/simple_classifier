import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

# Set the path to the folder containing the image datasets
data_path = './data/train_data'

# Define the image dimensions and other parameters
image_width, image_height = 224, 224
batch_size = 32
num_classes = 7

# Use the ImageDataGenerator to preprocess the images and generate batches of data
data_generator = ImageDataGenerator(rescale=1.0/255.0)
train_generator = data_generator.flow_from_directory(data_path,target_size=(image_width, image_height),batch_size=batch_size,class_mode='categorical')

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=6)

# Save the trained model
model.save('sl_model.h5')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'sl_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
