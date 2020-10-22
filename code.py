# data set has 600 training images (200 of planes, 200 of boats and 200 of cars) and 100 testing images (50 of planes, 50 of cars, 50 of boats)

import tensorflow as tf 
import keras
from keras.preprocessing.image import ImageDataGenerator 
import os
import numpy as np
import matplotlib.pyplot as plt

#data paths
train_data = 'v_data/train'
val_data = 'v_data/test'

train_samples = 600
val_samples = 150
epochs = 20
batch_size = 64

img_width = 224 
img_height = 224

# rescale images from 255
img_train_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,width_shift_range = 0.2,height_shift_range = 0.2,rotation_range = 40,zoom_range = 0.2,horizontal_flip = True)

train_datagen = img_train_gen.flow_from_directory(batch_size=batch_size,directory=train_data,shuffle=True, target_size=(img_width,img_height), class_mode='binary')

img_val_gen = ImageDataGenerator(rescale = 1./255)

val_datagen = img_val_gen.flow_from_directory(batch_size=batch_size,directory=val_data, shuffle=True, target_size=(img_width,img_height),class_mode='binary') 

# model,convolution layers with maxpooling
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)), tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Conv2D(64, (3,3), activation='relu'), tf.keras.layers.MaxPooling2D(2,2), tf.keras.layers.Conv2D(128, (3,3), activation='relu'), tf.keras.layers.MaxPooling2D(2,2), tf.keras.layers.Conv2D(128, (3,3), activation='relu'), tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Dropout(0.5),tf.keras.layers.Flatten(),tf.keras.layers.Dense(512, activation='relu'),tf.keras.layers.Dense(3, activation='softmax')])

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

history = model.fit_generator(train_datagen, steps_per_epoch=int(np.ceil(train_samples / float(batch_size))), epochs=epochs, validation_data=val_datagen, validation_steps=int(np.ceil(val_samples / float(batch_size))))

model.save('my_model.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
