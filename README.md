# Brain Tumor MRI Classification using Deep Learning (multi_model_comparison)

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



## 📓 Notebooks & Experimental Results

This project explores how different image enhancement techniques affect the classification of brain tumors. We evaluated three distinct approaches:

### 1. [Multi-Class Classification & Multi-Model Comparison](./notebooks/multi_class_classification_and_multi_model_comparison.ipynb)
* **Preprocessing:** **Histogram Equalization**
* **Description:** This is the primary notebook of the project. It applies Histogram Equalization to MRI scans to improve global contrast. This helps the models (EfficientNetB0, B3, B5) better identify tumor boundaries in low-contrast images.
* **Performance:** Achieved a **97% Weighted Average Accuracy** with EfficientNetB5.

### 2. [brain_tumor_classification-unsharpen_mask-multi](./notebooks/brain_tumor_classification-unsharpen_mask-multi.ipynb)
* **Preprocessing:** **Unsharp Masking**
* **Description:** This notebook uses an Unsharp Mask filter to sharpen the edges of the MRI scans. The goal was to test if enhancing fine structural details and edges would assist the Convolutional Neural Networks in feature extraction.
* **Performance:** High precision across most classes, particularly effective for identifying structural boundaries in complex tumor types.

### 3. [Baseline Model (No Preprocessing)](./notebooks/Brain%20Tumor%20MRI%20Classification_(without%20histogram%20equalization).ipynb)
* **Preprocessing:** **None (Baseline)**
* **Description:** This notebook serves as the control group. It uses standard MRI images without any advanced enhancement (No Histogram Equalization or Sharpening). 
* **Purpose:** To establish a baseline performance metric to measure the actual "gain" provided by image processing techniques.

---

## 📊 Summary of Findings
| Technique | Top Model | Weighted F1-Score | Key Observation |
| :--- | :--- | :--- | :--- |
| **Histogram Equalization** | EfficientNetB5 | **0.97** | Best overall contrast and accuracy. |
| **brain_tumor_classification-unsharpen_mask-multi** | EfficientNetB5 | 0.94 | Improved edge detection but slightly lower recall. |
| **No Preprocessing** | EfficientNetB5 | 0.94 | Strong baseline, but struggles with low-contrast scans. |



---
## 👥 Team Members

This project was a collaborative effort divided into specialized domains:

| Name | Domain & Responsibility | GitHub |
| :--- | :--- | :--- |
| **Fady_Bakheet** | **Image Preprocessing:** Technique selection & optimization | [@fady-bakheet](https://github.com/fady-bakheet) |
| **Gerges Marzouk** | **Image Preprocessing:** Dataset cleaning & enhancement | [@gerges-marzouk](https://github.com/gerges-marzouk) |
| **Abdullah Yasser** | **Model & Training:** Architecture selection (EfficientNet) | [@Abdo727](https://github.com/Abdo727) |
| **Mazen Omar** | **Post-Processing:** Result refinement & visualization | [@Mazen Omar](https://github.com/Mazen04om) |
| **Hussein Ebrahim** | **Results & Analysis:** Evaluation metrics & testing | [@Hussien2002](https://github.com/Hussien2002) |

# under supervision of:- Eng.Haidy
---

## 🛠️ Project Pipeline
The project follows a rigorous end-to-end pipeline:
1. **Preprocessing:** Grayscale conversion, Histogram Equalization, and Unsharp Masking.
2. **Modeling:** Transfer Learning using EfficientNet (B0, B3, B5).
3. **Training:** Fine-tuning with Adam optimizer and dynamic learning rates.
4. **Analysis:** Multi-class classification reports and confusion matrix generation.


---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/fady-bakheet/repository-name/issues).

## ⭐️ Show your support
Give a ⭐️ if this project helped you!
