import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pickle


class SalaryPredictor:
    """
    Enterprise-grade ML Pipeline for Salary Prediction.
    Handles Data Engineering, Database Export, and Automated Training.
    Algorithms: Linear Regression with K-Fold Cross Validation.
    """

    def __init__(self, csv_filename):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_dir = self.project_root / "data"
        self.csv_path = self.data_dir / csv_filename
        self.db_path = self.data_dir / "salary_data.db"
        self.model_path = self.data_dir / "salary_model.pkl"
        self.cols_path = self.data_dir / "model_columns.pkl"

        self.df = None
        self.final_df = None

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"dataset (csv) not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("data not loaded. Call load_csv() first.")

        # 1. Ordinal Encoding
        exp_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
        self.df["experience_rank"] = self.df["experience_level"].map(exp_map)

        # Grouping Job Title (Top 5)
        top_jobs = self.df["job_title"].value_counts().nlargest(5).index.tolist()
        self.df["job_group"] = self.df["job_title"].apply(lambda x: x if x in top_jobs else "Other")

        # NEW: Grouping Company Location (Top 4 + Other) -> The Golden Feature!
        top_locations = self.df["company_location"].value_counts().nlargest(4).index.tolist()
        self.df["location_group"] = self.df["company_location"].apply(lambda x: x if x in top_locations else "Other")

        # One-Hot Encoding (Added location_group)
        cat_cols = ["company_size", "employment_type", "remote_ratio", "job_group", "location_group"]
        self.df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)

        # Target Transformation
        self.df_encoded["log_salary"] = np.log1p(self.df_encoded["salary_in_usd"])

        # 6. Cleanup
        drop_cols = ["work_year", "salary", "salary_currency", "salary_in_usd", 
                     "employee_residence", "company_location", "job_title", "experience_level"]
        self.final_df = self.df_encoded.drop(columns=[c for c in drop_cols if c in self.df_encoded.columns])

    def prompt_save_to_db(self):
        if self.final_df is None:
            raise ValueError("data not preprocessed. Call preprocess_data() first.")

        user_input = (
            input("Do you want to save the cleaned data to DB? (yes/no): ")
            .strip()
            .lower()
        )

        if user_input in ["yes", "y", "غ"]:
            engine = create_engine(f"sqlite:///{self.db_path}")
            self.final_df.to_sql("cleaned_salaries", engine, if_exists="replace", index=False)
            print(f"Data saved successfully to {self.db_path.name}")
        else:
            print("Skipped database export")

    def train_and_evaluate(self):
        if not self.db_path.exists():
            raise FileNotFoundError("Database not found. Please run prompt_save_to_db() first.")

        # Load directly from Database
        engine = create_engine(f"sqlite:///{self.db_path}")
        df_db = pd.read_sql("SELECT * FROM cleaned_salaries", engine)

        cols_to_exclude = ["log_salary", "Unnamed: 0"]
        existing_exclude = [c for c in cols_to_exclude if c in df_db.columns]
        X = df_db.drop(columns=existing_exclude)
        Y = df_db["log_salary"]

        # K-Fold Cross Validation(shuffle = در هم و مخلوطی برش زدن)
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

        # Baseline Model (Linear Regression)
        print("\nEvaluating Baseline Model (Linear Regression)...")
        lr_model = LinearRegression()
        lr_scores = cross_val_score(lr_model, X, Y, cv=cv_strategy, scoring="r2")
        
        # Calculate real USD RMSE
        lr_log_pred = cross_val_predict(lr_model, X, Y, cv=cv_strategy)
        real_usd_y = np.expm1(Y)
        lr_rmse = np.sqrt(mean_squared_error(real_usd_y, np.expm1(lr_log_pred)))

        print(f"       LR Average R2: {lr_scores.mean():.4f}")
        print(f"       LR RMSE (USD): ${lr_rmse:,.0f}")

        # Random Forest model
        print("\n Evaluating Complex Model (Random Forest)...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_scores = cross_val_score(rf_model, X, Y, cv=cv_strategy, scoring="r2")
        
        rf_log_pred = cross_val_predict(rf_model, X, Y, cv=cv_strategy)
        rf_rmse = np.sqrt(mean_squared_error(real_usd_y, np.expm1(rf_log_pred)))

        print(f"       RF Average R2: {rf_scores.mean():.4f}")
        print(f"       RF RMSE (USD): ${rf_rmse:,.0f}")

        user_input = input("\nWhich model do you want to save? Enter 'lr', 'rf', or 'none': ").strip().lower()
        
        if user_input in ["lr", "rf"]:
            # select model by input user
            if user_input == "lr":
                champion_model = lr_model
                model_name = "Linear Regression"
            else:
                champion_model = rf_model
                model_name = "Random Forest"
                
            print(f"\nTraining {model_name} on the entire dataset...")
            champion_model.fit(X, Y)

            with open(self.model_path, "wb") as f:
                pickle.dump(champion_model, f)
            with open(self.cols_path, "wb") as f:
                pickle.dump(list(X.columns), f)
                
            print(f"Champion Model ({model_name}) successfully saved to {self.model_path.name}")
        else:
            print("Skipped model export.")


if __name__ == "__main__":
    app = SalaryPredictor("ds_salaries.csv")
    app.load_csv()
    app.preprocess_data()
    app.prompt_save_to_db()
    app.train_and_evaluate()