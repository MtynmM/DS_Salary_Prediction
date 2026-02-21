from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Initialize API App
app = FastAPI(title="ML Salary Predictor API", version="1.0.0")

# Load Model Artifacts (Pathlib for OS-independent paths)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "salary_model.pkl"
COLS_PATH = BASE_DIR / "data" / "model_columns.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(COLS_PATH, "rb") as f:
        model_columns = pickle.load(f)
    print("[INFO] Model and columns loaded successfully for Production.")
except Exception as e:
    model = None
    model_columns = None
    print(f"[ERROR] Failed to load model artifacts: {e}")

# Define Request Schema (Data Validation)
class SalaryRequest(BaseModel):
    experience_level: str  # (EN, MI, SE, EX)
    job_title: str         # e.g., Data Scientist 
    company_location: str  # e.g., US, IN, GB
    company_size: str      # S, M, L
    employment_type: str   # FT, PT
    remote_ratio: int      # 0, 50, 100

# 4. Expose Prediction Endpoint
@app.post("/predict")
def predict_salary(data: SalaryRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model is currently unavailable.")

    # Step A: Create an empty dictionary mapping all trained features to 0
    input_dict = {col: 0 for col in model_columns}

    # Step B: Inject Ordinal Feature
    exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
    input_dict["experience_rank"] = exp_map.get(data.experience_level, 2) # default MI

    # Step C: Dynamic One-Hot Encoding Injector
    def set_one_hot(prefix, value):
        exact_col = f"{prefix}_{value}"
        other_col = f"{prefix}_Other"
        
        if exact_col in input_dict:
            input_dict[exact_col] = 1
        elif other_col in input_dict:
            input_dict[other_col] = 1

    set_one_hot("job_group", data.job_title)
    set_one_hot("location_group", data.company_location)
    set_one_hot("company_size", data.company_size)
    set_one_hot("employment_type", data.employment_type)
    set_one_hot("remote_ratio", data.remote_ratio)

    # Step D: Execute Prediction
    df_input = pd.DataFrame([input_dict])
    log_pred = model.predict(df_input)[0]
    predicted_usd = np.expm1(log_pred)

    return {
        "predicted_salary_usd": round(predicted_usd, 0),
        "status": "success",
        "model_used": "Linear Regression"
    }