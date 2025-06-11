import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
IMG_SIZE = 256
CATEGORIES = ["Benign cases", "Malignant cases", "Normal cases", "Non_Pneumonia", "Pneumonia"]
MODEL_PATH = 'E:/mini project/save_model/keras_file/Lung_Cancer&Pneumonia_CNN_model[Version-2].keras'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
def preprocess_image(img_path):
    """Load and preprocess an image for classification."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Unable to read image {img_path}.")  
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_resized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    img_resized = np.repeat(img_resized, 3, axis=-1)  # Convert to 3 channels
    return img_resized
def classify_image(img_path):
    """Classify the uploaded image using the trained model."""
    img_processed = preprocess_image(img_path)
    prediction = model.predict(img_processed)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return CATEGORIES[class_index], confidence
def upload_and_classify():
    """Open a file dialog for the user to upload multiple images and classify them."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_paths:
        print("No files selected.")
        return
    for file_path in file_paths:
        try:
            class_name, confidence = classify_image(file_path)
            print(f"Image: {os.path.basename(file_path)}\nPredicted Class: {class_name} (Confidence: {confidence:.2f}%)\n")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    # Run the function to allow user to upload and classify multiple images
if __name__ == "__main__":
    upload_and_classify()

