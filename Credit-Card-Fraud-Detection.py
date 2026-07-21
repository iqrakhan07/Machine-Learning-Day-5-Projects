# Credit Card Fraud Detection using Logistic Regression

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Preview:\n")
print(df.head())

# Features and Target
X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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
print(confusion_matrix(y_test, y_pred))

# New Transaction
new_transaction = pd.DataFrame({
    "Time": [11],
    "Amount": [40000],
    "TransactionType": [0],
    "Location": [1]
})

prediction = model.predict(new_transaction)

print("\nNew Transaction Details")
print(new_transaction)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")