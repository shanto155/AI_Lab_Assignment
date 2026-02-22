import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = 'datasets/'  # Folder containing 'resizedShirt' and 'Ishirt' subfolders

# -----------------------------
# Dataset Loading
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='binary'   # Binary classification
)

test_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    subset='validation',
    class_mode='binary',
    shuffle=False
)

num_classes = 2  # Only two classes

# -----------------------------
# Build CNN Model
# -----------------------------
def build_model(filters1=32, filters2=64, filters3=128):
    model = Sequential([
        Conv2D(filters1, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),

        Conv2D(filters2, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(filters3, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# -----------------------------
# Training Time Measurement
# -----------------------------
start_train = time.time()
history = model.fit(train_data, epochs=EPOCHS)
end_train = time.time()
training_time = end_train - start_train
print("Total Training Time (seconds):", training_time)

# -----------------------------
# Testing Time per Sample
# -----------------------------
start_test = time.time()
loss, accuracy = model.evaluate(test_data)
end_test = time.time()
testing_time = end_test - start_test
time_per_sample = testing_time / test_data.samples

print("Test Accuracy:", accuracy)
print("Testing Time per Sample (seconds):", time_per_sample)

# -----------------------------
# Save Epoch vs Accuracy Plot
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.title("Epoch vs Accuracy / Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("epoch_vs_accuracy_binary.png", dpi=300)
plt.show()

# -----------------------------
# Optional: Data Size vs Performance
# -----------------------------
sizes = [0.25, 0.5, 0.75, 1.0]
accuracies = []

for s in sizes:
    train_subset = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='binary'
    )
    model_temp = build_model()
    model_temp.fit(train_subset, epochs=5, verbose=0)
    _, acc_temp = model_temp.evaluate(test_data, verbose=0)
    accuracies.append(acc_temp)

plt.figure()
plt.plot(sizes, accuracies)
plt.xlabel("Fraction of Training Data")
plt.ylabel("Accuracy")
plt.title("Data Size vs Accuracy")
plt.grid(True)
plt.savefig("data_vs_accuracy_binary.png", dpi=300)
plt.show()

# -----------------------------
# Optional: Model Size vs Performance
# -----------------------------
filter_configs = [(16,32,64), (32,64,128), (64,128,256)]
param_counts = []
model_accs = []

for f in filter_configs:
    m = build_model(*f)
    param_counts.append(m.count_params())
    m.fit(train_data, epochs=5, verbose=0)
    _, acc = m.evaluate(test_data, verbose=0)
    model_accs.append(acc)

plt.figure()
plt.plot(param_counts, model_accs)
plt.xlabel("Number of Parameters")
plt.ylabel("Accuracy")
plt.title("Model Size vs Accuracy")
plt.grid(True)
plt.savefig("modelsize_vs_accuracy_binary.png", dpi=300)
plt.show()
