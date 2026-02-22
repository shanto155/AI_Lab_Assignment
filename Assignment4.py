import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
import matplotlib.pyplot as plt
import os

# Make sure a folder exists to save plots
os.makedirs("plots", exist_ok=True)

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_dataset(dataset_name, dataset):
    (X_train, y_train), (X_test, y_test) = dataset

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Build model
    model = build_model(X_train.shape[1:])

    # Train model and capture history
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=2
    )

    # Evaluate test accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{dataset_name} Test Accuracy: {acc*100:.2f}%")

    # Plot training & validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{dataset_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = f"plots/{dataset_name.replace(' ', '_')}_accuracy.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}\n")

# Train and plot for each dataset
train_dataset("Fashion MNIST", fashion_mnist.load_data())
train_dataset("MNIST", mnist.load_data())
train_dataset("CIFAR-10", cifar10.load_data())
