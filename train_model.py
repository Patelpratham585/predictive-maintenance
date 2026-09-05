"""
Predictive Maintenance — Random Forest Classifier
AI4I 2020 Predictive Maintenance Dataset

Predicts 'Machine failure' from mechanical sensor readings:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]

NOTE ON LEAKAGE: the dataset also has TWF/HDF/PWF/OSF/RNF columns. These
record *which* failure mode occurred (tool wear failure, heat dissipation
failure, etc.), so they are only known AFTER a failure has happened. They
are intentionally excluded from the feature set — including them would let
the model "cheat" and give meaningless feature importances.
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
DATA_PATH = "ai4i2020.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"

X = df[FEATURES]
y = df[TARGET]

print(f"Dataset shape: {df.shape}")
print(f"Failure rate: {y.mean():.2%}  ({y.sum()} failures out of {len(y)} records)")

# ----------------------------------------------------------------------
# 2. Train / test split
#    stratify=y keeps the same ~3.4% failure ratio in both splits, since
#    failures are rare and a random split could easily under/over-sample them
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 3. Train the Random Forest
#    class_weight="balanced" tells the forest to weight the rare failure
#    class more heavily, so it doesn't just learn to always predict "no failure"
# ----------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. Evaluate on the held-out test set
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Evaluation on held-out test set (2,000 records) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(f"                 Predicted OK   Predicted Fail")
print(f"Actually OK      {cm[0][0]:<14} {cm[0][1]}")
print(f"Actually Fail    {cm[1][0]:<14} {cm[1][1]}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No failure", "Failure"]))

# ----------------------------------------------------------------------
# 5. Feature importance — which mechanical factor matters most?
# ----------------------------------------------------------------------
importances = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
)
print("\nFeature importances (higher = more predictive of failure):")
print(importances.to_string())

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.xlabel("Importance")
plt.title("What drives predicted machine failure?")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\nSaved chart to feature_importance.png")

# ----------------------------------------------------------------------
# 6. Save the trained model
# ----------------------------------------------------------------------
joblib.dump(model, "model.pkl")
print("Saved trained model to model.pkl")
