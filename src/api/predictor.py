from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT_PROJECT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_PROJECT / "data" / "artifact"


class Predictor():
    def __init__(self):
        """Load the model, feature columns, and category mappings from disk"""
        self.model = joblib.load(ARTIFACT_DIR / "lr.joblib")
        self.feature_columns = joblib.load(ARTIFACT_DIR / "lr_columns.joblib")
        mappings = joblib.load(ARTIFACT_DIR / "mappings.joblib")
        self.top_jobs = mappings["top_jobs"]
        self.top_locations = mappings["top_locations"]

    def _transform_input(self, raw_input: dict) -> pd.DataFrame:
        """
        Convert a raw input dictionary into a one-hot encoded DataFrame
        that exactly matches the training feature columns.
        """
        data = raw_input.copy()

        exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
        data["experience_rank"] = exp_map.get(data["experience_level"], 2) 

        data["job_group"] = data["job_title"] if data["job_title"] in self.top_jobs else "Other"

        data["location_group"] = data["company_location"] if data["company_location"] in self.top_locations else "Other"

        for col in ["experience_level", "job_title", "company_location"]:
            data.pop(col, None)

        df = pd.DataFrame([data])

        cat_cols = ["company_size", "employment_type", "remote_ratio", "job_group", "location_group"]
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        df_final = df_encoded.reindex(columns=self.feature_columns, fill_value=0)

        return df_final

    def predict(self, raw_input: dict) -> float:
        """
        Predict salary in USD from raw input.
        Returns the predicted annual salary (float).
        """
        X = self._transform_input(raw_input)
        log_pred = self.model.predict(X)[0]  
        salary_usd = np.expm1(log_pred)          
        return round(float(salary_usd), 2)