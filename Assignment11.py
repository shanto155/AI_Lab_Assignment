import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = 32  # Resize MNIST digits to 32x32
BATCH_SIZE = 128
EPOCHS = 5
os.makedirs("plots", exist_ok=True)

# -----------------------------
# Load and preprocess MNIST
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def preprocess_mnist(X):
    X_resized = np.stack([tf.image.resize(tf.expand_dims(img, -1), [IMG_SIZE, IMG_SIZE]).numpy() for img in X], axis=0)
    X_rgb = np.repeat(X_resized, 3, axis=-1)  # Convert grayscale to RGB
    X_rgb = X_rgb / 255.0
    return X_rgb

X_train_proc = preprocess_mnist(X_train)
X_test_proc = preprocess_mnist(X_test)

# -----------------------------
# Load pre-trained VGG16
# -----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Feature extraction only

# Add classification head for MNIST
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Transfer learning on MNIST
# -----------------------------
history = model.fit(X_train_proc, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_features(model, X):
    """
    Extract features from the Flatten layer and flatten for PCA/t-SNE.
    """
    feature_model = Model(inputs=model.input, outputs=model.layers[-3].output)  # Flatten layer
    features = feature_model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    num_samples = features.shape[0]
    features_flat = features.reshape(num_samples, -1)  # Flatten to (num_samples, H*W*C)
    return features_flat

# Features BEFORE transfer learning (VGG16 base only)
features_before = extract_features(Model(inputs=base_model.input, outputs=Flatten()(base_model.output)), X_test_proc)

# Features AFTER transfer learning
features_after = extract_features(model, X_test_proc)

# -----------------------------
# Dimensionality reduction & plotting
# -----------------------------
def plot_2d(features, labels, method='PCA', filename='plot.png'):
    """
    Reduce features to 2D using PCA or t-SNE and plot them.
    """
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    else:
        raise ValueError("Method must be 'PCA' or 'TSNE'")
    
    features_2d = reducer.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    for i in range(10):
        plt.scatter(features_2d[labels==i,0], features_2d[labels==i,1], label=str(i), alpha=0.6)
    plt.title(f"{method} Visualization of Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.show()
    print(f"Saved plot: plots/{filename}")

# -----------------------------
# Plot PCA and t-SNE BEFORE transfer learning
# -----------------------------
plot_2d(features_before, y_test, method='PCA', filename='features_before_PCA.png')
plot_2d(features_before, y_test, method='TSNE', filename='features_before_tSNE.png')

# -----------------------------
# Plot PCA and t-SNE AFTER transfer learning
# -----------------------------
plot_2d(features_after, y_test, method='PCA', filename='features_after_PCA.png')
plot_2d(features_after, y_test, method='TSNE', filename='features_after_tSNE.png')
