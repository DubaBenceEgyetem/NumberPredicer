import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import asarray

nine = 'pictures/nine.jpg'
eight = 'pictures/eight.jpg'
neweight = 'pictures/neweight.jpg'
five = 'pictures/five.jpg'

img_neweight = mpimg.imread(neweight)
img_nine = cv2.imread(nine)
img_eight = mpimg.imread(eight)

print("Shape of nine:", img_nine.shape) 
print("Shape of eight:", img_eight.shape)
print("Shape of neweight:", img_neweight.shape)


image = cv2.resize(img_nine, (28, 28))

# Kép normalizálása 0 és 1 közé
image = image / 255.0

image = np.reshape(image, (1, 28, 28, 3)) 
image = 1.0 - image  


plt.imshow(image[0])  
plt.savefig('newnine.jpg')
plt.show()
