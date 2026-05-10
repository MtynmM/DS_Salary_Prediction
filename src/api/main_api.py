# src/api/main_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

class SalaryInput(BaseModel):
    experience_level: str = Field(..., description="EN, MI, SE, or EX")
    employment_type: str = Field(..., description="FT, PT, CT, or FL")
    job_title: str = Field(..., description="e.g., Data Scientist, Machine Learning Engineer")
    remote_ratio: int = Field(..., description="0, 50, or 100")
    company_location: str = Field(..., description="ISO country code, e.g., US, GB, IN, DE")
    company_size: str = Field(..., description="S, M, or L")


def create_app(predictor):
    """
    Build and return a FastAPI app, injecting the predictor.
    """
    app = FastAPI(
        title="Salary Prediction API",
        description="Predict data science salary based on job features.",
        version="1.0.0"
    )

    @app.post("/predict")
    def predict_salary(input_data: SalaryInput):
        try:
            raw_dict = input_data.model_dump()
            predicted = predictor.predict(raw_dict)
            return {
                "predicted_salary_usd": predicted,
                "input": raw_dict
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    def root():
        return {"message": "Salary Prediction API is running."}

    return app