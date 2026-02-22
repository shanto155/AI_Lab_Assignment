import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load MNIST dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# One-hot encoding for MSE
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

# -----------------------------
# Experiment settings
# -----------------------------
activations = ['relu', 'tanh', 'sigmoid']
loss_functions = ['sparse_categorical_crossentropy', 'mean_squared_error']
results = {}

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# -----------------------------
# Model builder function
# -----------------------------
def build_model(activation='relu', loss_fn='sparse_categorical_crossentropy'):
    model = Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation=activation),
        Dropout(0.3),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

# -----------------------------
# Experiment 1: Activation Functions
# -----------------------------
for act in activations:
    print(f"\nTraining with activation: {act}")
    model = build_model(activation=act)
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        verbose=2
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[f"{act}_activation"] = {'history': history.history, 'test_acc': acc}
    
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title(f'Activation: {act}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/activation_{act}.png", dpi=300)
    plt.close()

# -----------------------------
# Experiment 2: Loss Functions
# -----------------------------
for loss_fn in loss_functions:
    print(f"\nTraining with loss function: {loss_fn}")
    model = build_model(loss_fn=loss_fn)
    
    # Use one-hot labels for MSE
    if loss_fn == 'mean_squared_error':
        y_train_input = y_train_onehot
        y_test_input = y_test_onehot
    else:
        y_train_input = y_train
        y_test_input = y_test

    history = model.fit(
        X_train, y_train_input,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        verbose=2
    )
    
    loss, acc = model.evaluate(X_test, y_test_input, verbose=0)
    results[f"{loss_fn}_loss"] = {'history': history.history, 'test_acc': acc}
    
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title(f'Loss Function: {loss_fn}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/loss_{loss_fn}.png", dpi=300)
    plt.close()

# -----------------------------
# Summary of Test Accuracies
# -----------------------------
print("\n=== Summary of Test Accuracies ===")
for key in results:
    print(f"{key}: {results[key]['test_acc']*100:.2f}%")
