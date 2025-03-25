# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using Machine Learning algorithms. It analyzes transaction patterns and classifies them as legitimate or fraudulent.

## Features
- Preprocessing and feature engineering on the dataset.
- Training multiple machine learning models.
- Evaluating performance metrics like accuracy, precision, recall, and F1-score.
- Predicting fraudulent transactions based on trained models.

## Algorithms Used
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Dataset
- The dataset consists of transaction details, including amount, time, and anonymized features.
- It includes labeled data for fraudulent (1) and non-fraudulent (0) transactions.
- Commonly used dataset: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Installation & Setup
1. Install required dependencies:
   ```sh
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. Load the dataset and preprocess it.
3. Train and evaluate models using the provided scripts.
4. Run predictions on new transactions.

## Model Evaluation
- **Confusion Matrix** to evaluate model performance.
- **ROC Curve & AUC Score** to measure classification effectiveness.
- **Precision-Recall Tradeoff** for handling imbalanced datasets.

## Future Enhancements
- Implement deep learning models like ANN or LSTM for improved detection.
- Optimize hyperparameters for better model accuracy.
- Deploy the model as an API for real-time fraud detection.


