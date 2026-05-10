from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import joblib


class DataPreparator:
    def __init__(self, csv_filename: str) -> None:
        """The file method creates an object from the path to make it faster and easier to use,
        and the resolve method converts the path from relative to absolute."""

        self.root_project = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.root_project / "data"
        self.csv_path = self.data_dir / "raw" / csv_filename
        self.db_path = self.data_dir / "db" / "salary_data.db"
        self.artifact_dir = self.data_dir / "artifact"

    def load_csv(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"dataset not found at: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

    def preparation(self) -> None:

        # Ordinal Encoding
        exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
        self.df["experience_rank"] = self.df["experience_level"].map(exp_map)

        # Grouping Job Title (Top 5)
        top_jobs = self.df["job_title"].value_counts().nlargest(5).index.tolist()
        self.df["job_group"] = self.df["job_title"].apply(
            lambda x: x if x in top_jobs else "Other"
        )
        self.top_jobs = top_jobs

        # Grouping Company Location (Top 4 + Other)
        top_locations = (
            self.df["company_location"].value_counts().nlargest(4).index.tolist()
        )
        self.df["location_group"] = self.df["company_location"].apply(
            lambda x: x if x in top_locations else "Other"
        )
        self.top_locations = top_locations 

        # One-Hot Encoding (Added location_group)
        cat_cols = [
            "company_size",
            "employment_type",
            "remote_ratio",
            "job_group",
            "location_group",
        ]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        # Target Transformation
        self.df_encoded["log_salary"] = np.log1p(self.df_encoded["salary_in_usd"])

        # Cleanup
        drop_cols = [
            "work_year",
            "salary",
            "salary_currency",
            "salary_in_usd",
            "employee_residence",
            "company_location",
            "job_title",
            "experience_level",
        ]
        self.final_df = self.df_encoded.drop(
            columns=[c for c in drop_cols if c in self.df_encoded.columns]
        )

    def save_to_db(self, table_name: str = "final_data") -> None:
        try:
            engine = create_engine(f"sqlite:///{self.db_path}")
            self.final_df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(
                f"Data saved successfully to table '{table_name}' in {self.db_path.name}"
            )
        except Exception as e:
            print(f"failed to save db: {e}")

    # run data_preparator(convert and customized df to db)
    def etl(self):
        self.load_csv()
        self.preparation()
        self.save_to_db()
        mappings = {
        "top_jobs": self.top_jobs,
        "top_locations": self.top_locations
        }
        joblib.dump(mappings, self.artifact_dir / "mappings.joblib")
        print('this is the End(DataPreparator)')
