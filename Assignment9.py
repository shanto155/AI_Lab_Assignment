import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.models import Model
import os

# -----------------------------
# Create folder for plots
# -----------------------------
os.makedirs("plots", exist_ok=True)

# -----------------------------
# Load and preprocess image
# -----------------------------
img_path = "image.jpg"  # Replace with your image path

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, C)

# -----------------------------
# Feature map visualization function
# -----------------------------
def visualize_feature_maps(model, preprocess_fn, save_name, max_channels=16):
    """
    model: pretrained CNN model
    preprocess_fn: preprocessing function (vgg_pre, res_pre, etc.)
    save_name: filename to save the plot
    max_channels: maximum number of channels to plot
    """
    # Preprocess the image
    img_processed = preprocess_fn(img_array.copy())

    # Select first convolutional layer
    conv_layers = [layer.output for layer in model.layers if "conv" in layer.name]
    if len(conv_layers) == 0:
        print(f"No convolution layers found in {model.name}")
        return

    layer_output = conv_layers[0]  # first conv layer only

    # Build feature extraction model
    feature_model = Model(inputs=model.input, outputs=layer_output)
    feature_maps = feature_model.predict(img_processed)

    fmap = feature_maps[0]  # shape: (H, W, C)
    num_channels = fmap.shape[-1]
    num_to_plot = min(max_channels, num_channels)

    # Plot feature maps
    plt.figure(figsize=(10, 10))
    for i in range(num_to_plot):
        plt.subplot(4, 4, i+1)
        plt.imshow(fmap[:, :, i], cmap='viridis')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/{save_name}", dpi=300)
    plt.show()
    print(f"Feature map plot saved as: plots/{save_name}")

# -----------------------------
# VGG16
# -----------------------------
vgg_model = VGG16(weights='imagenet', include_top=False)
visualize_feature_maps(vgg_model, vgg_pre, "vgg_feature_maps.png")

# -----------------------------
# ResNet50
# -----------------------------
resnet_model = ResNet50(weights='imagenet', include_top=False)
visualize_feature_maps(resnet_model, res_pre, "resnet_feature_maps.png")

# -----------------------------
# MobileNetV2
# -----------------------------
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False)
visualize_feature_maps(mobilenet_model, mob_pre, "mobilenet_feature_maps.png")
