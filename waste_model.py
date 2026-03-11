import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==============================
# Paths
# ==============================

train_dir = "entrainement"
test_dir = "test"

img_size = 128
batch_size = 16

# ==============================
# Data Augmentation
# ==============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ==============================
# Load Dataset
# ==============================

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# Transfer Learning Model
# ==============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# Train Model
# ==============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ==============================
# Evaluate Model
# ==============================

test_loss, test_acc = model.evaluate(test_data)

print("\nTest Accuracy:", test_acc)

# ==============================
# Confusion Matrix
# ==============================

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_data.classes, y_pred)

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_data.class_indices,
            yticklabels=test_data.class_indices)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# Save Model
# ==============================

model.save("waste_classifier_model.h5")

print("Model saved successfully!")