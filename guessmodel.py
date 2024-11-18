import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kapcsold be a logolást
#tf.debugging.set_log_device_placement(True)


print("Elérhető eszközök:", tf.config.list_physical_devices())



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train= X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


print(X_train.max())

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = tf.keras.Sequential([
  tf.keras.Input(shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss='categorical_crossentropy',
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
          metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Teszt veszteség: {test_loss}")
print(f"Teszt pontosság: {test_accuracy}")

model.save("saved_models/saved_model.h5")




'''
plt.plot(history.history['loss'], label='veszteségek')
plt.plot(history.history['val_loss'], label='Érvényességi veszteségek')
plt.xlabel('epochs')
plt.ylabel('veszteség')
plt.legend(),
plt.show()
'''

