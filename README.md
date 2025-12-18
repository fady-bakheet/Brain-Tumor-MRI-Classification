# Brain Tumor MRI Classification using Deep Learning

This repository features a comprehensive deep learning project for the multi-class classification of brain tumor MRI images. By utilizing **Transfer Learning** with several architectures from the **EfficientNet** family, the system achieves state-of-the-art precision in identifying 44 different tumor types.

## 🚀 Overview
* **Objective:** Automate the detection and classification of brain tumors to assist in diagnostic accuracy.
* **Dataset:** [Brain Tumor MRI Dataset (44 Classes)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) hosted on Kaggle.
* **Architecture:** Comparative analysis of **EfficientNetB0**, **EfficientNetB3**, and **EfficientNetB5**.

## 📊 Model Comparison & Results
The project evaluated three distinct models to find the optimal balance between computational cost and classification accuracy. All models were tested on a standard set of 448 images.

| Model | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1-Score |
| :--- | :---: | :---: | :---: |
| **EfficientNetB0** | 0.95 | 0.94 | 0.94 |
| **EfficientNetB3** | 0.96 | 0.96 | 0.96 |
| **EfficientNetB5** | **0.97** | **0.97** | **0.97** |

### Top Performer: EfficientNetB5
The **EfficientNetB5** model achieved exceptional results, particularly in critical tumor categories:
* **Glioblastoma:** 0.95 Precision / 1.00 Recall
* **Neurocitoma:** 1.00 Precision / 0.98 Recall
* **Ganglioglioma:** 1.00 Precision / 1.00 Recall
* **Germinoma:** 1.00 Precision / 1.00 Recall

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Frameworks:** TensorFlow, Keras
* **Libraries:** OpenCV, Scikit-learn, Matplotlib, Seaborn, Albumentations

## 🧬 Implementation Details
### 1. Data Preprocessing
Images are loaded and converted to grayscale to standardize the input. Normalization is applied to ensure efficient model convergence.

### 2. Data Augmentation
To prevent overfitting and improve robustness, `ImageDataGenerator` was used to apply real-time augmentations including:
* Rotation and Width/Height shifts.
* Horizontal flipping and Zooming.

### 3. Model Architecture
Each model utilized pre-trained **ImageNet** weights as a base. A custom classification head was added, consisting of:
* **BatchNormalization** for stable training.
* **Dense layers** with **L2 Regularization**.
* **Dropout (0.3)** to improve generalization.

### 4. Training Strategy
* **Optimizer:** Adam (Learning Rate: 0.001).
* **Callbacks:** `EarlyStopping` (patience: 5) and `ReduceLROnPlateau` to dynamically adjust learning rates during training plateaus.

## 📈 Visualizations
You can find the following plots within the `notebooks/` directory:
* **Image Processing:** Comparison of MRI scans before and after preprocessing.
* **Training History:** Accuracy and Loss curves for all three models.
* **Confusion Matrix:** Detailed breakdown of classification performance across all 44 classes.

## ⚙️ Setup and Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Brain-Tumor-Classification.git](https://github.com/your-username/Brain-Tumor-Classification.git)
