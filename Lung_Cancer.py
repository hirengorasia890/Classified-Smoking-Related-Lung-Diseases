import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")

# Define the main path to the dataset
directory = 'E:\mini project\datasets\lung_copy'

start_time = time.time()
# Categories and their names
# categories = ["Non_Pneumonia","Pneumonia"]
categories = ["Bengin cases","Malignant cases","Normal cases","Non_Pneumonia","Pneumonia"]

# Unified image size
img_size = 256

# Lists to store images and labels
data = []
labels = []

# Data exploration: Read images and inspect them
low_quality_images = []
dimensions = []

for category in categories:
    path = os.path.join(directory, category)
    class_num = categories.index(category)  # Assign a numeric label for each category

    for file in os.listdir(path):
        filepath = os.path.join(path, file)

        try:
            # Read the image in grayscale
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Record the dimensions of the image
            dimensions.append(img.shape)

            # Check the image quality
            brightness = np.mean(img)
            if brightness < 50:  # Threshold for low-quality images
                low_quality_images.append(filepath)
                continue

            # Resize the image
            img_resized = cv2.resize(img, (img_size, img_size))

            # Add the image and label to the lists
            data.append(img_resized)
            labels.append(class_num)
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

# Convert data to numpy arrays and normalize the images to [0, 1]
data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
labels = np.array(labels)
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# Split the data into 80% for training and 20% for testing (initially)
x_temp, x_test, y_temp, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Further split the remaining 80% into training (70%) and validation (10%)
x_train, x_valid, y_train, y_valid = train_test_split(x_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

# Handle class imbalance using SMOTE
print("Before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train.reshape(-1, img_size * img_size), y_train)
x_train_resampled = x_train_resampled.reshape(-1, img_size, img_size, 1)
print("After SMOTE:", Counter(y_train_resampled))


# Convert grayscale images to 3 channels (RGB)
x_train_resampled = np.repeat(x_train_resampled, 3, axis=-1)
x_valid = np.repeat(x_valid, 3, axis=-1)
x_test = np.repeat(x_test, 3, axis=-1)


# Build the CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten the data
model.add(Flatten())

# Add dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

model.add(Dense(5, activation='softmax'))  # 3 classes for classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Show the model summary
model.summary()
model.save('lung_model.h5')

# Callbacks for early stopping and model checkpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_cnn.keras', monitor='val_loss', save_best_only=True)
]

# Train the model
history = model.fit(
    x_train_resampled, y_train_resampled,
    validation_data=(x_valid, y_valid),
    epochs=25,
    batch_size=32,
    callbacks=callbacks
)

# Recompile the model with a smaller learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Fine-tune the model
history_fine_tune = model.fit(
    x_train_resampled, y_train_resampled,
    validation_data=(x_valid, y_valid),
    epochs=25,
    batch_size=32,
    callbacks=callbacks
)


# Function to show example images from each category
def show_images_for_categories(categories, directory, img_size):
    plt.figure(figsize=(15, 10))
    for i, category in enumerate(categories):
        path = os.path.join(directory, category)
        files = os.listdir(path)
        if files:
            img_path = os.path.join(path, files[0])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                plt.subplot(1, len(categories), i+1)
                plt.imshow(img, cmap='gray')
                plt.title(category)
                plt.axis('off')
    plt.suptitle("Sample Images from Each Category", fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to plot training and validation accuracy/loss
def plot_combined_training_history(history1, history2):
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training + Fine-Tuning Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training + Fine-Tuning Loss')

    plt.tight_layout()
    plt.show()


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.6f}")

# Calculate and print the training accuracy (accuracy on the training data)
train_loss, train_acc = model.evaluate(x_train_resampled, y_train_resampled)
print(f"Training Accuracy: {train_acc:.6f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_classes, target_names=categories)
print("Classification Report:\n", class_report)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"accuracy: {accuracy:.6f}")
# F1-Score
f1 = f1_score(y_test, y_pred_classes, average='weighted')
print(f"F1-Score: {f1:.6f}")

# Recall
recall = recall_score(y_test, y_pred_classes, average='weighted')
print(f"Recall: {recall:.6f}")

# Precision
precision = precision_score(y_test, y_pred_classes, average='weighted')
print(f"Precision: {precision:.6f}")

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
evaluation_results = {
    'confusion_matrix': conf_matrix,
    'classification_report': class_report,
    'accuracy': accuracy,
    'f1_score': f1,
    'recall': recall,
    'precision': precision
}
# Save the dictionary to a pickle file
# Call the function to display images


# Call the function to plot the training history
plot_combined_training_history(history, history_fine_tune)
with open('E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_CNN_model[Version-2]evaluation_results.pkl', 'wb') as f:
    pickle.dump(evaluation_results, f)
print("Results saved to 'Lung_Cancer&Pneumonia_CNN_model[Version-2]evaluation_results.pkl'")
joblib.dump(model, 'E:\mini project\save_model\joblib_file\Lung_Cancer&Pneumonia_CNN_model[Version-2].pkl')
model_save_path = 'E:\mini project\save_model\keras_file\Lung_Cancer&Pneumonia_CNN_model[Version-2].keras'
try:
    # Save the model
    model.save(model_save_path)

    # Check if the model is saved successfully
    if os.path.exists(model_save_path):
        print(f"Model has been saved successfully at {model_save_path}")
    else:
        print("Model saving failed.")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")
