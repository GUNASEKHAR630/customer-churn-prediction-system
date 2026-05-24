import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

print("Loading Telco Customer Churn Dataset")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] =df['TotalCharges'].fillna(df['TotalCharges'].median())

df['Churn'] = df['Churn'].map({'Yes' : 1, 'No': 0})

x = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = x.select_dtypes(include=['object']).columns
numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    le_dict[col] = le

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\nTraining samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")

print("\n Training Models...")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(x_train_scaled, y_train)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(x_train_scaled, y_train)
print("Models trained successfully")

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score:{auc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return y_pred, y_pred_proba

print("="*50)
print("Logistic Regression Results:")
lr_pred, lr_proba = evaluate_model(lr_model, x_test_scaled, y_test, "Logistic Regression")

print("\nRandom Forest Results:")
rf_pred, rf_proba = evaluate_model(rf_model, x_test_scaled, y_test, "random Forest")

joblib.dump(rf_model, 'churn_rf_model.pkl')
joblib.dump(scaler, 'churn_scaler.pkl')
joblib.dump(le_dict, 'churn_label_encoders.pkl')
print("\nBest model (Random Forest) and preprocessors saved")

def predict_churn(customer_data):
    model = joblib.load('churn_rf_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    encoders = joblib.load('churn_label_encoders.pkl')

    df_pred = pd.DataFrame([customer_data])

    required_cols = list(encoders.keys()) + ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']
    for col in required_cols:
        if col not in df_pred.columns:
            raise ValueError(f"Missing required column: {col}")

    df_pred['TotalCharges'] = pd.to_numeric(df_pred['TotalCharges'], errors='coerce')
    df_pred['TotalCharges'] = df_pred['TotalCharges'].fillna(df_pred['TotalCharges'].median())

    for col, le in encoders.items():
        if col in df_pred.columns:
            df_pred[col] = le.transform(df_pred[col])
    features_scaled = scaler.transform(df_pred)
    churn_prob = model.predict_proba(features_scaled)[0][1]
    prediction = "Yes" if churn_prob >= 0.5 else "No"

    return{
        "Churn_Prediction": prediction,
        'Churn_Probability': round(churn_prob * 100,2)   
    }

if __name__ == "__main__":
    print("\n"+ "="*60)
    print("CUSTOMER CHURN PREDICTION SYSTEM READY")
    print("="*60)

    example_customer = {
        'gender': 'Female',
        'SeniorCitizen':0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService' : 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup':'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract' :'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.6,
        'TotalCharges': 787.2
    }

    result = predict_churn(example_customer)
    print(result)
