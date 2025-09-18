#!/usr/bin/env python3

"""
Goal: Predict whether an individual has income >50K per year. 
    (Binary classification: 1 = >50K, 0 = ≤50K).

The train.py script will automatically:
- Standardize numeric features.
- One-hot encode categorical features.
- Fit a Logistic Regression classifier on top of this pipeline.

"""

# Import necessary libraries
import json, os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Define directories for data and models
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)  # Create models directory if it doesn't exist

# Load training and test datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Split features and labels for train and test sets
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]
X_test  = test_df.drop(columns=["label"])
y_test  = test_df["label"]

# Identify categorical and numeric columns
categorical_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

# Define preprocessing: scale numeric, one-hot encode categorical
"""ColumnTransformer allows you to apply different preprocessing steps to different columns in a dataset.
Each column type needs a different transformation before feeding it into a model like Logistic Regression.

Why scale numeric columns?
- Logistic Regression assumes features are on similar scales.
- If age ranges from 18–90 but capital_gain ranges 0–10000, the optimizer might be biased toward features with larger values.
- Gradient-based optimization is faster and more stable.
- with_mean=False is set here because:
    * Sometimes sparse matrices (from later one-hot encoding) can’t handle centering (subtracting the mean) efficiently.
    * Ensures the scaler works safely when combining numeric + one-hot categorical features.

Why one-hot encode categorical columns?
- Models like Logistic Regression require numeric inputs.
- Categorical strings (like "Private" or "Self-emp") can’t be directly processed by Logistic Regression.
- One-hot encoding creates a binary column for each category.
- handle_unknown='ignore' ensures:
    * If the model sees a category at inference time that wasn’t in training, it won’t crash.
    * Instead, it encodes it as all zeros.
"""
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), numeric_cols),  # Scale numeric columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # Encode categorical columns
    ]
)

# Initialize logistic regression classifier
clf = LogisticRegression(max_iter=300, n_jobs=None)

# Build pipeline: preprocessing + classifier
""" Why use a Pipeline + ColumnTransformer
- Encapsulates preprocessing + model in a single object:
    You don’t have to manually apply StandardScaler or OneHotEncoder every time.
- Safe for deployment:
    * Model can take raw CSV input, apply transformations, and produce predictions.
- Avoids data leakage:
    * Scaling and encoding are fit only on training data, then applied to test/new data automatically.
- Easier to maintain and extend:
    * Swap Logistic Regression for XGBoost, or add new features without rewriting preprocessing logic.
"""
pipe = Pipeline(steps=[('prep', preprocess), ('clf', clf)])

# Train the pipeline on the training data
pipe.fit(X_train, y_train)

# Evaluate the model on the test set
proba = pipe.predict_proba(X_test)[:, 1]  # Get predicted probabilities for positive class
auc = roc_auc_score(y_test, proba)  # Compute ROC AUC score
report = classification_report(y_test, (proba >= 0.5).astype(int), output_dict=True)  # Classification report

# Save metrics to JSON file
metrics = {"roc_auc": float(auc), "report": report}
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save trained pipeline to disk
joblib.dump(pipe, os.path.join(MODELS_DIR, "model.joblib"))

# Save schema (column types) to JSON for future reference
with open(os.path.join(MODELS_DIR, "schema.json"), "w") as f:
    json.dump({"numeric": numeric_cols, "categorical": categorical_cols}, f, indent=2)

# Print summary to console
print("Saved models/model.joblib and models/metrics.json")
print(f"ROC AUC: {auc:.3f}")