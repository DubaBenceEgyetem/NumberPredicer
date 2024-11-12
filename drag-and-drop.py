import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
import cv2

# Kapcsold be a logolást
#tf.debugging.set_log_device_placement(True)


print("Elérhető eszközök:", tf.config.list_physical_devices())



(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train_norm = X_train / 255.0, 
X_test_norm = X_test / 255.0


#y_train = to_categorical(y_train, num_classes=10)
#y_test = to_categorical(y_test, num_classes=10)


print(f"itt van ami kell {X_train.dtype, X_train.shape}") 
print(f"itt van ami kell {X_test.dtype, X_test.shape}") 
print(f"itt van ami kell {y_train.dtype, y_train.shape}") 
print(f"itt van ami kell {y_test.dtype, y_test.shape}") 


model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(10, activation="sigmoid")
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
          metrics=["accuracy"])


history = model.fit(X_train_norm, y_train, epochs=5, batch_size=32, validation_data=(X_test_norm, y_test))
test_loss, test_accuracy = model.evaluate(X_test_norm, y_test)
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

