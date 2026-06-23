# Automated Diagnosis of Lung Diseases Linked to Smoking via CNNs and SMOTE-Enhanced Imaging Data

This repository contains the code for the research paper published at **IEEE ASPCON 2025**.

> **Published:** November 2025  
> **DOI:** [10.1109/ASPCON66877.2025.11389330](https://doi.org/10.1109/ASPCON66877.2025.11389330)

---

## What This Project Does

Lung diseases linked to smoking — lung cancer (benign and malignant) and pneumonia — are notoriously hard to distinguish from chest X-rays alone. This project trains and compares three CNN architectures on grayscale lung X-ray images to automate that classification. A major challenge was severe class imbalance in the dataset; we handled it using SMOTE (Synthetic Minority Oversampling Technique) before training.

Three models were tested: a custom CNN built from scratch, MobileNetV2 (transfer learning), and ResNet152 (transfer learning). All three were trained on 256×256 grayscale images converted to 3-channel RGB for compatibility with pretrained ImageNet weights.

---

## Dataset

The dataset has five classes:

- Benign cases
- Malignant cases
- Normal cases
- Non-Pneumonia
- Pneumonia

Images are loaded in grayscale, filtered for quality (brightness threshold < 50 discarded), resized to 256×256, and normalized to [0, 1]. The 80/10/10 train/validation/test split is stratified to preserve class ratios.

SMOTE runs on the flattened training images before reshaping back to 256×256×3. This generates synthetic minority-class samples so no single class dominates training.


---

## Project Structure

```
classified-smoking-related-lung-diseases/
│
├── Lung_Cancer.py                  # Custom CNN model (built from scratch)
├── MobileNet_Model.py              # Transfer learning with MobileNetV2
├── ResNet_Model.py                 # Transfer learning with ResNet50
├── ResNet150_Model.py              # Transfer learning with ResNet152
├── Compare_Model.py                # Loads saved pickle results and plots a bar chart comparing all three models
├── classify_Lung_Cancer.py         # GUI-based single-image classifier using the trained CNN
├── LungCancer_Pneumonia_Predictin.py  # GUI-based classifier for the pneumonia binary model
└── README.md
```

---

## Models

### Custom CNN (`Lung_Cancer.py`)
Three convolutional blocks (32 → 64 → 128 filters), each followed by MaxPooling. Flattened into a 128-unit dense layer with 0.5 dropout, then a 5-class softmax output. Trained for up to 25 epochs, then fine-tuned at a lower learning rate (0.0001).

### MobileNetV2 (`MobileNet_Model.py`)
Uses pretrained ImageNet weights with the base frozen. A GlobalAveragePooling layer feeds into a 128-unit dense layer, dropout, and softmax output. Trained for 30 epochs, fine-tuned for 25.

### ResNet50 (`ResNet_Model.py`) and ResNet152 (`ResNet150_Model.py`)
Same architecture as MobileNetV2 but with ResNet50 and ResNet152 bases respectively.

All models use:
- Adam optimizer
- Sparse categorical crossentropy loss
- EarlyStopping (patience=5, restores best weights)
- ModelCheckpoint (saves best `.keras` file)

---

## Evaluation

Each training script prints and saves:
- Confusion matrix
- Classification report (per-class precision, recall, F1)
- Weighted accuracy, F1, recall, precision
- Training execution time

Results are saved as `.pkl` files. `Compare_Model.py` loads all three pickle files and produces a grouped bar chart comparing accuracy, precision, recall, and F1 across models.

---

## Running the Code

### Requirements

```bash
pip install tensorflow opencv-python scikit-learn imbalanced-learn matplotlib joblib
```

### Training

Update the `directory` variable at the top of each script to point to your dataset folder, then run:

```bash
python Lung_Cancer.py        # Train custom CNN
python MobileNet_Model.py    # Train MobileNetV2
python ResNet150_Model.py    # Train ResNet152
```

### Comparing Models

After training all three, update the pickle file paths in `Compare_Model.py` and run:

```bash
python Compare_Model.py
```

### Single Image Classification

```bash
python classify_Lung_Cancer.py
```

A file dialog opens. Select one or more `.jpg`/`.png` images. The script loads the saved CNN model, preprocesses each image, and prints the predicted class with confidence.

---

## Known Issues / Notes

- All file paths are absolute Windows paths. Linux/macOS users need to update them.
- `Compare_Model.py` contains mostly commented-out legacy code; only the bottom section (pickle loading + bar chart) is active.
- `LungCancer_Pneumonia_Predictin.py` uses a separate binary pneumonia model not included in this repo. Update `model_path` to point to your saved `.keras` file.
- Model files (`.keras`, `.pkl`, `.h5`) are not included in this repo due to size. Train the models first.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{aspcon2025lungcnn,
  title     = {Automated Diagnosis of Lung Diseases Linked to Smoking via CNNs and SMOTE-Enhanced Imaging Data},
  booktitle = {2025 IEEE Asia Symposium on Power and Computing (ASPCON)},
  year      = {2025},
  month     = {November},
  doi       = {10.1109/ASPCON66877.2025.11389330}
}
```

---

## License

This project is released for academic and research use. See the paper for full methodological details.
