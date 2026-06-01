import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

print("Loading dataset...")

df = pd.read_csv("data/application_train.csv")

print("Dataset Shape:", df.shape)

# ---------------------------------------
# REMOVE ID COLUMN
# ---------------------------------------

if "SK_ID_CURR" in df.columns:
    df = df.drop("SK_ID_CURR", axis=1)

# ---------------------------------------
# TARGET / FEATURES
# ---------------------------------------

y = df["TARGET"]

X = df.drop("TARGET", axis=1)

# ---------------------------------------
# PREPROCESSING
# ---------------------------------------

print("Preprocessing data...")

label_encoders = {}
median_values = {}

for col in X.columns:

    if X[col].dtype == "object":

        X[col] = X[col].fillna("Unknown")

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        label_encoders[col] = le

    else:

        median_values[col] = X[col].median()

        X[col] = X[col].fillna(
            median_values[col]
        )

print("Preprocessing completed")

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ---------------------------------------
# MODEL
# ---------------------------------------

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    class_weight="balanced",
    random_state=42
)

print("Training LightGBM model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully")

# ---------------------------------------
# PREDICTIONS
# ---------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]
# Feature Importance

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features:\n")

print(feature_importance.head(20))

# ---------------------------------------
# METRICS
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)

print("ROC AUC  :", roc_auc)

# ---------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/credit_risk_model.pkl"
)

joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

joblib.dump(
    median_values,
    "models/median_values.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

print("\nArtifacts Saved:")

print("✔ credit_risk_model.pkl")

print("✔ label_encoders.pkl")

print("✔ median_values.pkl")

print("✔ feature_columns.pkl")