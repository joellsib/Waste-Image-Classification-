import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("waste_classifier_model.h5")

classes = ['glass','metallic','paper','plastic']

# Load test image
img = cv2.imread("test.jpg/img4.jpeg")
img = cv2.resize(img,(128,128))
img = img/255.0

img = np.expand_dims(img,axis=0)

prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Predicted Waste Type:", classes[class_index])