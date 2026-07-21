# Credit Risk Prediction using Logistic Regression

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")

# Display Dataset
print("First 5 Rows:\n")
print(df.head())

# Features and Target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keeps class distribution balanced
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
print("Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

# New Customer Prediction
new_customer = pd.DataFrame({
    "Age": [30],
    "Income": [50000],
    "LoanAmount": [18000],
    "CreditScore": [690],
    "EmploymentYears": [5]
})

prediction = model.predict(new_customer)

print("\nNew Customer Details")
print(new_customer)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("Low Credit Risk")
else:
    print("High Credit Risk")