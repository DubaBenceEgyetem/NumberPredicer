import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os 


model = tf.keras.models.load_model("saved_models/saved_model.h5")
model.summary()

path = 'pictures'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for i in range(len(files)):
    image = cv2.imread(f'pictures/{files[i]}', cv2.IMREAD_GRAYSCALE)
    print("Kép alakja:", image.shape)
    image = cv2.resize(image, (28, 28))
    image_norm = image / 255.0
    image = np.reshape(image_norm, (1, 28, 28, 1))
    image = 1.0 - image
    
    print("Átalakított kép alakja és típusa:", np.shape(image), image.dtype)

    predictnumber = model.predict(image)
    predicted_class = np.argmax(predictnumber, axis=1)
    print(predictnumber, predicted_class)
    print(f"A képen látható szám: {predicted_class[0]}")
    print(f"Ami valójában a képen volt: {files[i]}")    
    plt.imshow(image.reshape( 28, 28), cmap='gray')
    plt.show()   


