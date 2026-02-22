# Assignment 2
# Fully Connected Feed-forward Neural Network Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create model
model = Sequential()

# Input layer (6 features) → Hidden Layer 1 (12 neurons)
model.add(Dense(12, activation='relu', input_shape=(6,)))

# Hidden Layer 2 (8 neurons)
model.add(Dense(8, activation='relu'))

# Output Layer (3 classes)
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print summary
model.summary()

# Save model architecture image
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='Problem_2.png', show_shapes=True)

print("Model successfully built.")
