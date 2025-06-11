import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tkinter import filedialog
from tkinter import Tk

model_path = r"E:\6-Semester\MiniProject\Models\Save_Model\Pneumonia_model[1].keras" 
# Check if the model files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file for pneumonia not found: {model_path}")

# Load models
Pneumonia_Disease_Model = tf.keras.models.load_model(model_path)
# Define labels for pneumonia model
Pneumonia_Disease_labels = ["Non_Pneumonia", "Pneumonia"]
Pneumonia_Disease_target_size = (256, 256)
# Function to preprocess an image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Function to classify pneumonia or Pneumonia cancer
def classify_disease(img_path):
    img_array = preprocess_image(img_path, Pneumonia_Disease_target_size)

    # Get predictions from the pneumonia model
    Pneumonia_Disease_pred = Pneumonia_Disease_Model.predict(img_array)

    # Apply softmax to get the probabilities (if the model is outputting raw logits)
    Pneumonia_Disease_pred_prob = tf.nn.softmax(Pneumonia_Disease_pred).numpy()
    print(Pneumonia_Disease_pred_prob)

    # Get the class with the highest probability
    Pneumonia_Disease_class = np.argmax(Pneumonia_Disease_pred_prob, axis=1)[0]  # [0] to get the scalar value
    print(Pneumonia_Disease_class)
    return Pneumonia_Disease_labels[Pneumonia_Disease_class]

# Function to get the image path using a file dialog
def select_image_paths():
    # Set up Tkinter window
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_paths

# Test with local images
img_paths = select_image_paths()  # Let the user select the image(s)

# Classify each selected image
for img_path in img_paths:
    disease_class = classify_disease(img_path)
    print(f"{img_path}: ({disease_class})")


