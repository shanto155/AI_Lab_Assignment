import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Generate data
x = np.linspace(-10, 10, 1000)
x = x.reshape(-1,1)

# Choose equation
def linear(x):
    return 5*x + 10

def quadratic(x):
    return 3*x**2 + 5*x + 10

def cubic(x):
    return 4*x**3 + 3*x**2 + 5*x + 10

y = quadratic(x)   # Change to linear(x) or quadratic(x)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=32,
                    verbose=1)

# Predict
y_pred = model.predict(X_test)

# Plot
plt.scatter(X_test, y_test, label="Original")
plt.scatter(X_test, y_pred, label="Predicted")
plt.legend()
plt.title("Original vs Predicted")
plt.savefig("Problem_3_quadratic.png")
plt.show()

print("Test Loss:", model.evaluate(X_test, y_test))
