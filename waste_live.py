import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("waste_classifier_model.h5")

classes = ['glass','metallic','paper','plastic']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    label = classes[class_index]

    # Show prediction on screen
    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Waste Classifier", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()