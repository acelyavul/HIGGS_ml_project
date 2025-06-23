# HIGGS Boson Classification: Model Comparison Using ROC-AUC

## 1. Problem Definition  
In this study, classification models were developed and compared using the HIGGS Boson dataset to distinguish between positive and negative classes.  
The goal is to evaluate the discriminatory power of the models, particularly using **ROC curves** and **AUC scores**.

## 2. Methodology

- **Dataset**: UCI HIGGS (sample size: 100,000 rows)
- **Preprocessing**:
  - Winsorizing (based on IQR)
  - MinMax Scaling
- **Feature Selection**:
  - ANOVA F-test (top 15 features selected)
- **Models Used**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Multi-layer Perceptron (MLP)
  - Extreme Gradient Boosting (XGBoost)
- **Validation Strategy**:
  - Nested Cross-Validation  
    - Outer loop: 5-fold  
    - Inner loop: 3-fold
- **Performance Metric**:
  - ROC-AUC (Receiver Operating Characteristic – Area Under Curve)

---

This setup aims to identify the most effective classifier by evaluating and comparing performance under a consistent validation strategy.
