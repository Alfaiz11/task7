# Task 7
# Breast Cancer Classification using Support Vector Machines (SVM)

This project applies Support Vector Machine (SVM) models to classify tumors as malignant or benign using the [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset). The aim is to explore SVM with both linear and RBF kernels, visualize decision boundaries, tune hyperparameters, and evaluate performance using cross-validation.

---

## Dataset

- **Source**: Kaggle ([Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset))
- **Target variable**: `diagnosis` (M = Malignant, B = Benign)
- **Features**: 30 numeric features derived from images of cell nuclei
- **Size**: ~570 samples

---

## Project Tasks

### 1. Data Preprocessing
- Load the dataset with pandas.
- Drop unnecessary columns (`id`, `Unnamed: 32`).
- Encode the target variable (`M` = 1, `B` = 0).
- Split into training and testing sets.
- Standardize features using `StandardScaler`.

### 2. Train SVM Classifiers
- Train an SVM with a **linear kernel**.
- Train another SVM with a **Radial Basis Function (RBF) kernel**.
- Evaluate both using classification reports.

### 3. Visualize Decision Boundaries
- Use PCA to reduce features to 2D.
- Visualize the decision boundary for RBF SVM.

### 4. Hyperparameter Tuning
- Use `GridSearchCV` to find the best values of:
  - `C` (regularization)
  - `gamma` (kernel coefficient)
- Optimize model performance with cross-validation.

### 5. Cross-Validation Evaluation
- Perform 10-fold cross-validation using `cross_val_score`.
- Report average accuracy and standard deviation.

---

## Results

- **Best SVM Parameters (from Grid Search)**: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
- **Cross-validation Accuracy**: Mean Accuracy ~91.38%

