Heart Disease Prediction
Overview
This repository contains a machine learning pipeline for predicting heart disease. It utilizes various classification algorithms to process medical data, scale numerical features, and evaluate model performance to find the most accurate predictor.

Datasets
The project is built to handle two primary datasets:

Kaggle Dataset: Defaults to the johnsmith88/heart-disease-dataset, which includes 14 Cleveland-like attributes and a target column.

UCI Hungarian Dataset: Uses the raw hungarian.data file. The pipeline is configured to parse 14 standard attributes, treat -9 as NaN, drop missing values, and binarize the target column (num).

Models Evaluated
The following machine learning models are trained and compared in this project:

Logistic Regression

Linear Support Vector Machine (SVM)

Random Forest Classifier (Identified as the best model in the default run)

XGBoost

Evaluation Metrics
The models are evaluated using a comprehensive suite of metrics:

Accuracy, Precision, Recall, and F1-Score

ROC-AUC Score and ROC Curves for visual comparison

Confusion Matrix

Feature Importance (specifically extracted for tree-based models)

Data Preprocessing Pipeline
Feature Separation: Automatically separates numerical and categorical columns.

Scaling: Applies StandardScaler to numerical features within a Pipeline and ColumnTransformer.

Data Split: Splits the dataset into 80% training data and 20% testing data, utilizing stratified sampling to maintain class balance.

Installation & Prerequisites
To run this notebook, you will need the following dependencies installed:

pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, kaggle.

Kaggle API Setup (Optional but recommended)
Place your Kaggle API token (kaggle.json) at ~/.kaggle/kaggle.json (ensure chmod 600 on Linux/macOS).

If the Kaggle download fails or the API is not set up, the script includes a fallback mechanism to download the heart.csv dataset from alternative GitHub mirror URLs.

Usage Notes
To switch between datasets, change the DATASET_CHOICE variable in the notebook to either "heart" or "uci_hungarian".

If using the UCI Hungarian dataset, ensure the hungarian.data file is placed in the data/ directory (or /content/ if running on Google Colab).

The pipeline currently drops categorical columns by default, but you can easily add categorical encoders (like OneHotEncoder) to the categorical_transformer pipeline if you introduce new categorical features.
