import pandas as pd
import shap
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/application_train.csv")

# Remove ID column
df = df.drop("SK_ID_CURR", axis=1)

# Separate target
y = df["TARGET"]
X = df.drop("TARGET", axis=1)

# Same preprocessing as training
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown")

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    else:
        X[col] = X[col].fillna(X[col].median())

print("Data preprocessing completed")

# Load trained model
model = joblib.load("credit_risk_model.pkl")

print("Model loaded")

# Use small sample for faster SHAP computation
sample = X.sample(1000, random_state=42)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(sample)

# Summary Plot
shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

print("SHAP Summary Plot saved as shap_summary.png")