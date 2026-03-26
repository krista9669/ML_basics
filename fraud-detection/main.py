# Supervised Learning -> Binary Classification
# Logistic Regression is a classification algorithm

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

target = "is_fraud"

X_train = train_data.drop(target, axis=1)   # all columns except is_fraud
y_train = train_data[target]    # only is_fraud

X_test = test_data.drop(target, axis=1)
y_test = test_data[target]

# logistic regression only works with numbers hence the filteration since the other information could cause errors
X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test = X_test.select_dtypes(include=["int64", "float64"])

# as learnt previously scaling is done here
scaler = StandardScaler()   
X_train = scaler.fit_transform(X_train) # fitted -> learning the pattern and scaled (transformed)
X_test = scaler.transform(X_test)   # only transform since fit can cause data leakage

model = LogisticRegression(max_iter=1000, class_weight="balanced")
# training is done through iteration -> The model is allowed to try up to 1000 steps to learn
# class_weight="balanced" -> catch more fraud cases and recall better
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))