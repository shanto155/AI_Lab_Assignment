import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import os

# -----------------------------
# Parameters
# -----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = 'datasets/'  # Folder containing 'resizedShirt' and 'Ishirt'

os.makedirs("plots", exist_ok=True)

# -----------------------------
# Data Preparation
# -----------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='binary'
)

# -----------------------------
# Build VGG16-based Model
# -----------------------------
def build_vgg16_model(trainable_layers='all'):
    """
    trainable_layers: 'all' for full fine-tuning
                      'top' for partial fine-tuning (freeze lower layers)
    """
    # Load VGG16 base
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    if trainable_layers == 'top':
        # Freeze lower layers, only last 4 convolutional blocks trainable
        for layer in base_model.layers[:-4*3]:  # Approx 4 conv blocks = 4*3 layers
            layer.trainable = False
    else:
        # Whole model trainable
        for layer in base_model.layers:
            layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Training Function
# -----------------------------
def train_and_plot(trainable_type):
    if trainable_type == 'all':
        model_name = "vgg16_whole_finetune"
        print("\n=== Training: Whole VGG16 Fine-tuning ===")
    else:
        model_name = "vgg16_partial_finetune"
        print("\n=== Training: Partial VGG16 Fine-tuning ===")
    
    model = build_vgg16_model(trainable_layers=trainable_type)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=2
    )

    # Evaluate test accuracy on validation set
    loss, acc = model.evaluate(val_data, verbose=0)
    print(f"{model_name} Test Accuracy: {acc*100:.2f}%")

    # Plot training & validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plot_path = f"plots/{model_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Plot saved to {plot_path}\n")

    return acc

# -----------------------------
# Run Both Fine-tuning Strategies
# -----------------------------
acc_whole = train_and_plot('all')
acc_partial = train_and_plot('top')

print("Final Test Accuracy:")
print(f"Whole VGG16 Fine-tuning: {acc_whole*100:.2f}%")
print(f"Partial VGG16 Fine-tuning: {acc_partial*100:.2f}%")

