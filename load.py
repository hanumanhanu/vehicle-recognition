import tensorflow as tf

import cv2

import numpy as np
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

img = cv2.imread('v_data/train/planes/1.jpg')

img = cv2.resize(img,(224,224))

img = np.reshape(img, (1,224,224,3))

classes = model.predict_classes(img)

print(classes)
