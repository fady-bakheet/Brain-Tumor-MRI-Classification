# Brain Tumor MRI Classification using EfficientNet

This repository contains a deep learning project for multi-class classification of brain tumor MRI images. By leveraging transfer learning with the EfficientNet family, the models achieve high precision and recall across 44 distinct tumor categories.

## 🚀 Overview
* **Objective:** Automate detection and classification of brain tumors from MRI scans to assist in medical diagnostics.
* **Dataset:** [Brain Tumor MRI Dataset (44 Classes)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) hosted on Kaggle.
* **Architecture:** Evaluated EfficientNetB0 through EfficientNetB5.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, Scikit-image
* **Metrics:** Scikit-learn (Classification Reports, Confusion Matrices)

## 📊 Results (EfficientNetB5)
The **EfficientNetB5** model demonstrated superior performance on the test set:
* **Weighted Average Precision:** 0.97
* **Weighted Average Recall:** 0.97
* **Weighted Average F1-Score:** 0.97

### Key Class Performance:
| Tumor Type | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Glioblastoma** | 0.95 | 1.00 | 0.98 |
| **Neurocitoma** | 1.00 | 0.98 | 0.99 |
| **Ganglioglioma** | 1.00 | 1.00 | 1.00 |

## 🧬 Methodology
1. **Preprocessing:** Grayscale conversion and image normalization.
2. **Data Augmentation:** Used `ImageDataGenerator` for rotations and scaling to improve generalization.
3. **Model Fine-tuning:** Custom classification heads added to pre-trained EfficientNet weights.
4. **Optimization:** Implemented `EarlyStopping` and `ReduceLROnPlateau` for efficient training.
