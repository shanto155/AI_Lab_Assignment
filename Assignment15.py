import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
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
# Callbacks
# -----------------------------
os.makedirs("callbacks_models", exist_ok=True)

callbacks = [
    # Stop training if validation loss does not improve for 3 epochs
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
    
    # Save the best model during training
    ModelCheckpoint(filepath='callbacks_models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    
    # Log training history to CSV file
    CSVLogger('callbacks_models/training_log.csv', append=False)
]

# -----------------------------
# Train the model with callbacks
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# -----------------------------
# Evaluate the model
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
