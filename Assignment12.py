import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Use all CPU cores
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# -----------------------------
# Load Dataset (Reduced Size)
# -----------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reduce dataset size for faster CPU training
X_train = X_train[:20000]
y_train = y_train[:20000]

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# Smaller CNN Model
# -----------------------------
def build_model():
    model = Sequential([
        Input(shape=(32,32,3)),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

EPOCHS = 5
BATCH = 128

# -----------------------------
# 1️⃣ Without Augmentation
# -----------------------------
model_no_aug = build_model()
history_no_aug = model_no_aug.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    verbose=1
)
val_acc_no_aug = history_no_aug.history['val_accuracy'][-1]

# -----------------------------
# 2️⃣ Flip Augmentation
# -----------------------------
flip_gen = ImageDataGenerator(horizontal_flip=True, validation_split=0.2)

model_flip = build_model()
history_flip = model_flip.fit(
    flip_gen.flow(X_train, y_train, batch_size=BATCH, subset='training'),
    epochs=EPOCHS,
    validation_data=flip_gen.flow(X_train, y_train, batch_size=BATCH, subset='validation'),
    verbose=1
)
val_acc_flip = history_flip.history['val_accuracy'][-1]

# -----------------------------
# 3️⃣ Rotation Augmentation
# -----------------------------
rot_gen = ImageDataGenerator(rotation_range=15, validation_split=0.2)

model_rot = build_model()
history_rot = model_rot.fit(
    rot_gen.flow(X_train, y_train, batch_size=BATCH, subset='training'),
    epochs=EPOCHS,
    validation_data=rot_gen.flow(X_train, y_train, batch_size=BATCH, subset='validation'),
    verbose=1
)
val_acc_rot = history_rot.history['val_accuracy'][-1]

# -----------------------------
# 4️⃣ Combined Augmentation
# -----------------------------
combo_gen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

model_combo = build_model()
history_combo = model_combo.fit(
    combo_gen.flow(X_train, y_train, batch_size=BATCH, subset='training'),
    epochs=EPOCHS,
    validation_data=combo_gen.flow(X_train, y_train, batch_size=BATCH, subset='validation'),
    verbose=1
)
val_acc_combo = history_combo.history['val_accuracy'][-1]

# -----------------------------
# Compare Results
# -----------------------------
methods = ["No Aug", "Flip", "Rotation", "Combined"]
accuracies = [val_acc_no_aug, val_acc_flip, val_acc_rot, val_acc_combo]

plt.figure()
plt.bar(methods, accuracies)
plt.ylabel("Validation Accuracy")
plt.title("Effect of Data Augmentation")
plt.savefig("augmentation_comparison.png", dpi=300)
plt.show()

print("\nValidation Accuracies:")
for m, a in zip(methods, accuracies):
    print(f"{m}: {a:.4f}")
