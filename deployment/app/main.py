from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Union
import pandas as pd
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.joblib")
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Income >50K Predictor", version="1.0")

class CensusRecord(BaseModel):
    age: int
    workclass: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str] = None

def _predict_df(df: pd.DataFrame):
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return [{"probability_over_50k": float(p), "prediction": int(c)} for p, c in zip(proba, pred)]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Union[CensusRecord, List[CensusRecord]]):
    if isinstance(item, list):
        df = pd.DataFrame([r.model_dump() for r in item])
    else:
        df = pd.DataFrame([item.model_dump()])
    # Minimal NA handling: replace None with 'Unknown' for strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)
    return _predict_df(df)
