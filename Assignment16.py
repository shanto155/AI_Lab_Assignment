import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

# -----------------------------
# Prepare MNIST dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# -----------------------------
# Build simple FCFNN model
# -----------------------------
def build_model():
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# -----------------------------
# Train the model and save history
# -----------------------------
os.makedirs("plots", exist_ok=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=128,
    verbose=2
)

# -----------------------------
# Plot Accuracy Curve
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('plots/accuracy_curve.png', dpi=300)
plt.show()

# -----------------------------
# Plot Loss Curve
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('plots/loss_curve.png', dpi=300)
plt.show()

# -----------------------------
# Evaluate the model
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
