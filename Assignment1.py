# Assignment 1
# Fully Connected Feed-forward Neural Network
# Architecture: 8 -> 4 -> 8 -> 4 -> 10

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the model
model = Sequential()

# Input layer (8 features) -> Hidden Layer 1 (4 neurons)
model.add(Dense(4, activation='relu', input_shape=(8,)))

# Hidden Layer 2 (8 neurons)
model.add(Dense(8, activation='relu'))

# Hidden Layer 3 (4 neurons)
model.add(Dense(4, activation='relu'))

# Output Layer (10 neurons)
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Optional: Save model architecture image
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='Problem_1.png', show_shapes=True)

print("\nModel successfully created.")
