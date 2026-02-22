import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
import matplotlib.pyplot as plt
import os

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_plot(dataset_name, dataset, is_cifar=False):
    (X_train, y_train), (X_test, y_test) = dataset

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape for CNN
    if not is_cifar:
        X_train = X_train[..., tf.newaxis]
        X_test = X_test[..., tf.newaxis]

    model = build_cnn(X_train.shape[1:])
    
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
    
    # Plot accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{dataset_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plot_path = f"plots/{dataset_name.replace(' ', '_')}_Problem_5.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}\n")

# Train on datasets
train_and_plot("Fashion MNIST", fashion_mnist.load_data())
train_and_plot("MNIST", mnist.load_data())
train_and_plot("CIFAR-10", cifar10.load_data(), is_cifar=True)
