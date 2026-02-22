import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Use all CPU cores
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# -----------------------------
# Load Dataset (Reduced Size)
# -----------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reduce dataset for CPU speed
X_train = X_train[:25000]
y_train = y_train[:25000]

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

EPOCHS = 10
BATCH = 128

# -----------------------------
# Model Builder (Smaller Model)
# -----------------------------
def build_model(use_dropout=False):
    model = Sequential([
        Input(shape=(32,32,3)),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu')
    ])

    if use_dropout:
        model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -----------------------------
# Model A: No Dropout, No Aug
# -----------------------------
model_A = build_model(False)
history_A = model_A.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Model B: Dropout Only
# -----------------------------
model_B = build_model(True)
history_B = model_B.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Augmentation Generator
# -----------------------------
aug_gen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    validation_split=0.2
)

# -----------------------------
# Model C: Augmentation Only
# -----------------------------
model_C = build_model(False)
history_C = model_C.fit(
    aug_gen.flow(X_train, y_train, batch_size=BATCH, subset='training'),
    epochs=EPOCHS,
    validation_data=aug_gen.flow(X_train, y_train, batch_size=BATCH, subset='validation'),
    verbose=1
)

# -----------------------------
# Model D: Dropout + Aug
# -----------------------------
model_D = build_model(True)
history_D = model_D.fit(
    aug_gen.flow(X_train, y_train, batch_size=BATCH, subset='training'),
    epochs=EPOCHS,
    validation_data=aug_gen.flow(X_train, y_train, batch_size=BATCH, subset='validation'),
    verbose=1
)

# -----------------------------
# Plot Comparison
# -----------------------------
plt.figure()
plt.plot(history_A.history['val_accuracy'])
plt.plot(history_B.history['val_accuracy'])
plt.plot(history_C.history['val_accuracy'])
plt.plot(history_D.history['val_accuracy'])

plt.legend(["No Reg", "Dropout", "Augmentation", "Dropout+Aug"])
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Effect on Overfitting")
plt.savefig("overfitting_accuracy.png", dpi=300)
plt.show()

