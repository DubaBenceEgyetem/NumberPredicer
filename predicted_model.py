import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2

model = tf.keras.models.load_model("saved_models/saved_model.h5")
model.summary()


image = cv2.imread('neweight.jpg', cv2.IMREAD_GRAYSCALE)
print("Kép alakja:", image.shape)
image = cv2.resize(image, (28, 28))
image_norm = image / 255.0
image = np.reshape(image_norm, (1, 28, 28, 1))
print("Átalakított kép alakja és típusa:", np.shape(image), image.dtype)

predictnumber = model.predict(image)
predicted_class = np.argmax(predictnumber, axis=1)
predicted_class[:10]
print(predictnumber, predicted_class)
print(f"A képen látható szám: {predicted_class[0]}")



plt.imshow(image.reshape( 28, 28), cmap='gray')
plt.show()