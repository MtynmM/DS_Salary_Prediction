import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine


class SalaryPredictor:
    """
    Enterprise-grade ML Pipeline for Salary Prediction.
    Handles Data Engineering, Database Export, and Automated Training.
    Algorithms: Linear Regression with K-Fold Cross Validation.
    """

    def __init__(self, csv_filename):
        self.project_root = Path(__file__).resolve().parent.parent
        self.csv_path = self.project_root / "data" / csv_filename
        self.db_path = self.project_root / "data" / "salary_data.db"

        self.df = None
        self.final_df = None

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"dataset (csv) not found at: {self.csv_path}")
        
        self.df= pd.read_csv(self.csv_path)

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("data not loaded. Call load_csv() first.")
        
        #ordinal encoding
        exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
        self.df['experience_rank']=self.df['experience_level'].map(exp_map)

        #grouping job title
        top_jobs = self.df["job_title"].value_counts().nlargest(5).index.tolist()
        self.df["job_group"] = self.df["job_title"].apply(lambda x: x if x in top_jobs else "Other")

        #one hot encoding
        cat_cols = ["company_size", "employment_type", "remote_ratio", "job_group"]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        #target transformation(log)
        self.df_encoded["log_salary"] = np.log1p(self.df_encoded["salary_in_usd"])

        #cleanup
        drop_cols = ["work_year", "salary", "salary_currency", "salary_in_usd", 
                     "employee_residence", "company_location", "job_title", "experience_level"]
        self.final_df = self.df_encoded.drop(columns=[c for c in drop_cols if c in self.df_encoded.columns])

    def prompt_save_to_db(self):
        if self.final_df is None:
            raise ValueError("data not preprocessed. Call preprocess_data() first.")
        
        user_input = input("Do you want to save the cleaned data to DB? (yes/no): ").strip().lower()

        if user_input in ['yes', 'y']:
            engine = create_engine(f"sqlite:///{self.db_path}")
            # storing data without log column for easier use
            db_df = self.final_df.drop(columns=['log_salary'], errors='ignore')
            db_df.to_sql('cleaned_salaries', engine, if_exists='replace', index=False)
            print(f"Data saved successfully to {self.db_path.name}")
        else:
            print("Skipped database export")

if __name__ == "__main__":
    app = SalaryPredictor("ds_salaries.csv")
    app.load_csv()
    app.preprocess_data()
    app.prompt_save_to_db()