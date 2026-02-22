import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load MNIST Dataset
# --------------------------------------------------
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

# Normalize
X_train_mnist = X_train_mnist / 255.0
X_test_mnist = X_test_mnist / 255.0

# --------------------------------------------------
# 2. Create "Your Custom Handwritten Dataset"
# (Using part of MNIST to simulate collected data)
# --------------------------------------------------
X_custom = X_train_mnist[:5000]
y_custom = y_train_mnist[:5000]

# Split into 80% train and 20% test
X_custom_train, X_custom_test, y_custom_train, y_custom_test = train_test_split(
    X_custom, y_custom, test_size=0.2, random_state=42
)

print("Custom Training Samples:", X_custom_train.shape[0])
print("Custom Testing Samples:", X_custom_test.shape[0])

# --------------------------------------------------
# 3. Combine Custom Training with MNIST Training
# --------------------------------------------------
X_combined_train = np.concatenate((X_train_mnist, X_custom_train), axis=0)
y_combined_train = np.concatenate((y_train_mnist, y_custom_train), axis=0)

print("Total Combined Training Samples:", X_combined_train.shape[0])

# --------------------------------------------------
# 4. Build FCFNN Model
# --------------------------------------------------
model = Sequential([
    Flatten(input_shape=(28, 28)),
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

model.summary()

# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------
history = model.fit(
    X_combined_train,
    y_combined_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2
)

# --------------------------------------------------
# 6. Evaluate on Custom Test Set
# --------------------------------------------------
custom_loss, custom_acc = model.evaluate(X_custom_test, y_custom_test)
print("\nAccuracy on Custom Test Dataset: {:.2f}%".format(custom_acc * 100))

# --------------------------------------------------
# 7. Evaluate on MNIST Test Set
# --------------------------------------------------
mnist_loss, mnist_acc = model.evaluate(X_test_mnist, y_test_mnist)
print("Accuracy on MNIST Test Dataset: {:.2f}%".format(mnist_acc * 100))

# --------------------------------------------------
# 8. Plot and Save Accuracy Curve
# --------------------------------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 9. Plot and Save Loss Curve
# --------------------------------------------------
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig("loss_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# 10. Plot and Save Custom vs MNIST Accuracy Comparison
# --------------------------------------------------
custom_accuracy = custom_acc * 100
mnist_accuracy = mnist_acc * 100

plt.figure()
plt.bar(['Custom Dataset', 'MNIST Dataset'],
        [custom_accuracy, mnist_accuracy])

plt.ylabel('Accuracy (%)')
plt.title('Custom vs MNIST Test Accuracy')

# Add value labels
plt.text(0, custom_accuracy + 0.5, f"{custom_accuracy:.2f}%")
plt.text(1, mnist_accuracy + 0.5, f"{mnist_accuracy:.2f}%")

plt.savefig("dataset_accuracy_comparison.png", dpi=300)
plt.show()

print("\nFigures Saved:")
print("1. accuracy_plot.png")
print("2. loss_plot.png")
print("3. dataset_accuracy_comparison.png")
