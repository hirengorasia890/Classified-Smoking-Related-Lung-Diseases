# import joblib
# import numpy as np
# import os
# from tensorflow.keras.models import load_model
# import pickle
# import matplotlib.pyplot as plt
# model2 = load_model(r'E:\mini project\save_model\keras_file\Lung_Cancer&Pneumonia_model_MobileNet[Version-1].keras')
# # Recommended change (use raw string or forward slashes)
# model1 = load_model(r'E:\mini project\save_model\keras_file\Lung_Cancer&Pneumonia_CNN_model[Version-2].keras')
# model3 = load_model(r'E:\mini project\save_model\keras_file\Lung_Cancer&Pneumonia_model_ResNet_Model[Version-1].keras')
 




# # # # Assuming you saved performance metrics as JSON, CSV, or pickle

# # # with open('model1_metrics.json', 'r') as f:
# # #     model1_metrics = json.load(f)

# # # with open('model2_metrics.json', 'r') as f:
# # #     model2_metrics = json.load(f)
# # # with open('model2_metrics.json', 'r') as f:
# # #     model3_metrics = json.load(f)

# # # print(f"Model 1 Metrics: {model1_metrics}")
# # # print(f"Model 2 Metrics: {model2_metrics}")


# # # model1_weights = model1.get_weights()
# # # model2_weights = model2.get_weights()
# # # # model3_weights = model3.get_weights()


# # # weight_diff = np.sum([np.sum(np.abs(w1 - w2)) for w1, w2 in zip(model1_weights, model2_weights)])
# # # print(f"Difference in weights: {weight_diff}")


# # # print(f"Model 1 size: {os.path.getsize('model1.h5')} bytes")
# # # print(f"Model 2 size: {os.path.getsize('model2.h5')} bytes")



# # Load the pickle file to retrieve the results
# with open('E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_CNN_model[Version-2]evaluation_results.pkl', 'rb') as f:
#     loaded_results = pickle.load(f)

# # Access the stored data
# print("Lung_Cancer&Pneumonia_model[Version-1]")
# model1.summary()
# print("CNN_Model Evaluation Results")
# print("Confusion Matrix:")
# print(loaded_results['confusion_matrix'])
# print("\nClassification Report:")
# print(loaded_results['classification_report'])
# print(f"\nAccuracy: {loaded_results['accuracy']:.6f}")
# print(f"F1-Score: {loaded_results['f1_score']:.6f}")
# print(f"Recall: {loaded_results['recall']:.6f}")
# print(f"Precision: {loaded_results['precision']:.6f}")




# with open('E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_model_MobileNet[Version-2]evaluation_results.pkl', 'rb') as f:
#     loaded_results = pickle.load(f)

# # Access the stored data
# print("\n")
# print("\n")
# print("\n")
# print("Lung_Cancer&Pneumonia_mobilenet")
# model2.summary()
# print("MobileNet_Model Evaluation Results")
# print("Confusion Matrix:")
# print(loaded_results['confusion_matrix'])
# print("\nClassification Report:")
# print(loaded_results['classification_report'])
# print(f"\nAccuracy: {loaded_results['accuracy']:.6f}")
# print(f"F1-Score: {loaded_results['f1_score']:.6f}")
# print(f"Recall: {loaded_results['recall']:.6f}")
# print(f"Precision: {loaded_results['precision']:.6f}")


# with open('E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_model_ResNet_Model[Version-1]evaluation_results.pkl', 'rb') as f:
#     loaded_results = pickle.load(f)

# # # Access the stored data
# print("\n")
# print("\n")
# print("\n")
# print("Lung_Cancer&Pneumonia_model_ResNet_Model[Version-1]")
# model3.summary()
# print("ResNet_Model Evaluation Results")
# print("Confusion Matrix:")
# print(loaded_results['confusion_matrix'])
# print("\nClassification Report:")
# print(loaded_results['classification_report'])
# print(f"\nAccuracy: {loaded_results['accuracy']:.6f}")
# print(f"F1-Score: {loaded_results['f1_score']:.6f}")
# print(f"Recall: {loaded_results['recall']:.6f}")
# print(f"Precision: {loaded_results['precision']:.6f}")


# Paths to the pickle files
import pickle
import matplotlib.pyplot as plt

# File paths
file_paths = {
    'CNN': r'E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_CNN_model[Version-2]evaluation_results.pkl',
    'MobileNet': r'E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_model_MobileNet[Version-2]evaluation_results.pkl',
    'ResNet152': r'E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_model_ResNet_Model[Version-1]evaluation_results.pkl'
}

# Collect metrics
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

# Load and store metrics
for model_name, file_path in file_paths.items():
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
        metrics['Model'].append(model_name)
        metrics['Accuracy'].append(results['accuracy'])
        metrics['Precision'].append(results['precision'])
        metrics['Recall'].append(results['recall'])
        metrics['F1-Score'].append(results['f1_score'])

# Plotting
x = range(len(metrics['Model']))
bar_width = 0.2

plt.figure(figsize=(12, 8))

# Plot each metric
bars_acc = plt.bar([i - 1.5 * bar_width for i in x], metrics['Accuracy'], width=bar_width, label='Accuracy')
bars_prec = plt.bar([i - 0.5 * bar_width for i in x], metrics['Precision'], width=bar_width, label='Precision')
bars_recall = plt.bar([i + 0.5 * bar_width for i in x], metrics['Recall'], width=bar_width, label='Recall')
bars_f1 = plt.bar([i + 1.5 * bar_width for i in x], metrics['F1-Score'], width=bar_width, label='F1-Score')

# Add values on top of bars (formatted to .6f)
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.6f}', ha='center', va='bottom', fontsize=14,rotation=20, fontweight='bold')

# Chart labels
plt.xticks(x, metrics['Model'],fontweight='semibold', fontsize=20)
plt.xlabel("Model",fontweight='semibold', fontsize=20)
plt.ylabel("Score",fontweight='semibold', fontsize=20)
plt.title("Evaluation Metrics Comparison",fontweight='semibold')
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
add_labels(bars_acc)
add_labels(bars_prec)
add_labels(bars_recall)
add_labels(bars_f1)

plt.tight_layout()


# Show plot
plt.show()




# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the stored evaluation results
# with open('E:\mini project\save_model\pickle_file\Lung_Cancer&Pneumonia_CNN_model[Version-2]evaluation_results.pkl', 'rb') as f:
#     loaded_results = pickle.load(f)

# # Access the stored data
# print("\n\n\n")
# print("CNN_Model Evaluation Results")
# print("Confusion Matrix:")
# print(loaded_results['confusion_matrix'])

# print("\nClassification Report:")
# print(loaded_results['classification_report'])

# print(f"\nAccuracy: {loaded_results['accuracy']:.2f}")
# print(f"F1-Score: {loaded_results['f1_score']:.2f}")
# print(f"Recall: {loaded_results['recall']:.2f}")
# print(f"Precision: {loaded_results['precision']:.2f}")

# # Function to plot confusion matrix
# def plot_confusion_matrix(cm, class_labels):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_labels,
#                 yticklabels=class_labels)

#     plt.xlabel('Predicted Labels')
#     plt.ylabel('True Labels')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.show()

# # Plot using the loaded confusion matrix
# plot_confusion_matrix(loaded_results['confusion_matrix'], [
#     "Benign cases", "Malignant cases", "Normal cases", "Non_Pneumonia", "Pneumonia"
# ])
