import joblib, pandas as pd

model = joblib.load("models/model.joblib")
sample = pd.read_csv("data/test.csv").drop(columns=["label"]).iloc[:2]
preds = model.predict(sample)
probs = model.predict_proba(sample)[:, 1]
print("Predictions:", preds)
print("Probabilities:", probs)
