# heart_disease_prediction

## Overview

This project aims to predict the possibility of heart disease in a patient based on clinical and medical parameters. Using a dataset derived from the Cleveland database (UCI ML Repository), we explore various machine learning models, including **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Random Forest Classifier**, to identify the best model for predicting heart disease. The goal is to maximize accuracy through techniques like hyperparameter tuning, feature selection, cross-validation, and evaluation metrics.

## Problem Definition

The challenge is to predict the likelihood of a patient having heart disease based on their medical attributes. Given clinical data about the physical condition of a patient, can we predict whether they have heart disease or not?

## Dataset

The dataset used in this project is sourced from the Cleveland heart disease dataset in the UCI ML Repository. The dataset consists of 76 attributes, but we focus on a subset of 14 key features. The target variable, **`goal`**, indicates the presence of heart disease, with values ranging from **0** (no heart disease) to **4** (presence of heart disease). For this project, we simplify the classification to distinguish between the presence (values 1,2,3,4) and absence (value 0) of heart disease.

### Data Source:
- **Cleveland Heart Disease Dataset**: [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Features:

- **age**: Age of the patient (in years)
- **sex**: Gender (1 = male; 0 = female)
- **cp**: Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar (> 120 mg/dl) (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0: Nothing to note, 1: ST-T Wave abnormality, 2: Left ventricular hypertrophy)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0: Upsloping, 1: Flatsloping, 2: Downsloping)
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalium stress result (1,3: normal, 6: fixed defect, 7: reversible defect)
- **target**: Presence of heart disease (1 = yes, 0 = no) – target variable

## Evaluation

We aim to reach a model accuracy of **95%** in predicting whether a patient has heart disease based on the features provided. The evaluation metrics used include:

- **Accuracy Score**: Percentage of correct predictions.
- **Cross-Validation**: To ensure the model generalizes well to unseen data.
- **Hyperparameter Tuning**: Randomized Search CV and GridSearch CV to optimize model performance.

## Approach

### 1. **Data Preprocessing**:
   - Handling missing values and categorical encoding.
   - Feature scaling for distance-based models (KNN).
   - Data splitting into training and testing sets.

### 2. **Modeling**:
   - **Logistic Regression**: A linear model used for binary classification.
   - **K-Nearest Neighbors (KNN)**: A non-parametric method for classification based on the distance between data points.
   - **Random Forest Classifier**: An ensemble method that constructs multiple decision trees for classification.

### 3. **Hyperparameter Tuning**:
   - **RandomizedSearchCV**: Randomized search over hyperparameters to find the best-performing ones.
   - **GridSearchCV**: Exhaustive search over a specified parameter grid.

### 4. **Model Evaluation**:
   - Evaluating the models using accuracy, precision, recall, F1-score, and cross-validation.
   - Comparison of performance to select the best model.

### 5. **Experimentation**:
   - Various models and tuning techniques are tested to determine the best accuracy for heart disease prediction.

## Results

The following accuracy scores were achieved for each model after hyperparameter tuning and validation:

- **Logistic Regression**: 88.52%
- **K-Nearest Neighbors (KNN)**: 68.85%
- **Random Forest Classifier**: 83.61%

Based on these results, **Logistic Regression** performed the best with an accuracy of **88.52%**.


1. Clone the repository:
   ```bash
   git clone https://github.com/VishakhViswanath/heart_disease_prediction.git

2.Conclusion

This project demonstrates the use of machine learning models to predict the likelihood of heart disease in patients based on their medical attributes. By testing Logistic Regression, K-Nearest Neighbors, and Random Forest Classifier, we found that Logistic Regression achieved the highest accuracy of 88.52%, making it the most reliable model for heart disease prediction in this dataset. Future work may include trying additional models and refining the feature selection process.

3.Acknowledgements

    Dataset provided by the UCI Machine Learning Repository.
    Libraries used: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
