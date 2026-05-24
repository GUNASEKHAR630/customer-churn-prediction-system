# Customer Churn Prediction System

This project predicts whether a customer is likely to leave a company (churn) using Machine Learning techniques in Python. The project demonstrates an end-to-end Machine Learning workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

## Features
- Data Cleaning and Preprocessing
- Handling Missing Values
- Encoding Categorical Variables
- Feature Scaling
- Exploratory Data Analysis (EDA)
- Train-Test Split
- Machine Learning Model Training
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation and Visualization
- Customer Churn Prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Concepts Used
- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- OneHotEncoder
- RandomForestClassifier
- LogisticRegression
- GridSearchCV
- Confusion Matrix

## Dataset
The dataset contains customer-related information such as:
- Gender
- Age
- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Contract Type
- Payment Method

Target Variable:
- Churn (0 = No, 1 = Yes)

## Project Workflow
1. Load Dataset
2. Perform Data Cleaning
3. Handle Missing Values
4. Perform Exploratory Data Analysis
5. Encode Categorical Features
6. Scale Numerical Features
7. Split Dataset into Training and Testing Data
8. Train Machine Learning Model
9. Tune Hyperparameters using GridSearchCV
10. Evaluate Model Performance
11. Predict Customer Churn

## Output
The model predicts whether a customer will churn and displays:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Best Hyperparameters

## How to Run

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
