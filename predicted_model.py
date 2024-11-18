import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
import os 


model = tf.keras.models.load_model("saved_models/saved_model.h5")
model.summary()
images_to_plot = []

path = 'pictures'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

for i in range(len(files)):
    image = cv2.imread(f'pictures/{files[i]}',cv2.IMREAD_GRAYSCALE)
    print("Kép alakja:", image.shape)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 28, 28))
    image = 1.0 - image
    
    print("Átalakított kép alakja és típusa:", np.shape(image), image.dtype)

    predictnumber = model.predict(image)
    predicted_class = np.argmax(predictnumber, axis=1)
    print(predictnumber, predicted_class)
    print(f"A képen látható szám: {predicted_class[0]}")
    print(f"Ami valójában a képen volt: {files[i]}")    



    images_to_plot.append((image, files[i], predicted_class[0]))



num_images = len(images_to_plot)
cols = 3  
rows = (num_images + cols - 1) // cols   
f, axarr = plt.subplots(rows, cols, figsize=(12,12))
axarr = axarr.flatten()

for i, (img, filename, pred) in enumerate(images_to_plot):
    axarr[i].imshow(img.reshape(28, 28), cmap='gray')
    axarr[i].set_title(f'File: {filename}, Predicted: {pred}')
    axarr[i].axis('off')
plt.tight_layout()
plt.show()

    


