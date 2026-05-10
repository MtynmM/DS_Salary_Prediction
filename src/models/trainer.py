from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib


#parents[2] same parent.parent.parent
ROOT_PROJECT = Path(__file__).resolve().parents[2] 
DATA_DIR = ROOT_PROJECT / "data"
DB_PATH = DATA_DIR / "db" / "salary_data.db"
MODEL_DIR = DATA_DIR / "artifact"


class ModelTrainer():
    def __init__(self, table_name : str = 'final_data'):
        self.table_name = table_name

    def load_data(self) -> pd.DataFrame:
        engine = create_engine(f'sqlite:///{DB_PATH}')
        query = f"SELECT * FROM {self.table_name}"
        df = pd.read_sql(query, engine)
        return df

    def prepare_features_target(self, df : pd.DataFrame):
        cols_to_drop = ['log_salary', "Unnamed: 0"]
        existing_exclude = [i for i in cols_to_drop if i in df]
        X = df.drop(columns=existing_exclude)
        Y = df['log_salary']
        return X, Y

    def evaluate_models(self,  X: pd.DataFrame, Y: pd.Series, cv_split: int = 5):
        """
        Evaluate Linear Regression and Random Forest using K-Fold Cross-Validation
        Print R² and RMSE (in real USD) for comparison
        Returns a dictionary with metrics only (no model objects)
        """
        cv = KFold(cv_split, shuffle=True, random_state= 42)
        y_real_usd = np.expm1(Y)

        # Linear Regression
        lr = LinearRegression()
        lr_r2_scores = cross_val_score(lr, X, Y, cv=cv, scoring="r2")
        lr_pred_log = cross_val_predict(lr, X, Y, cv=cv)
        lr_pred_real = np.expm1(lr_pred_log)
        #Root mean square error
        lr_rmse = np.sqrt(mean_squared_error(y_real_usd, lr_pred_real))

        # Random Forest
        rf = RandomForestRegressor(
            # number of trees
            n_estimators=100,
            # limit tree depth to prevent overfitting and save data to memories
            max_depth=10,
            # use all CPU cores
            random_state= 42, n_jobs=-1 
            )
        rf_r2_scores = cross_val_score(rf, X, Y, cv=cv, scoring="r2")
        rf_pred_log = cross_val_predict(rf, X, Y, cv=cv)
        rf_pred_usd = np.expm1(rf_pred_log)
        #Root mean square error, sqrt is √
        rf_rmse = np.sqrt(mean_squared_error(y_real_usd, rf_pred_usd))

        results = {
        "lr": {
            "r2_mean": lr_r2_scores.mean(),
            "r2_std":  lr_r2_scores.std(),
            "rmse_usd": lr_rmse
        },
        "rf": {
            "r2_mean": rf_r2_scores.mean(),
            "r2_std":  rf_r2_scores.std(),
            "rmse_usd": rf_rmse
        }
    }

        print(" Cross-Validation Results (5 folds):")
        # :.4f = چهار رقم اعشاری
        print(f"   Linear Regression | R² = {lr_r2_scores.mean():.4f} ±{lr_r2_scores.std():.4f} | RMSE = ${lr_rmse:,.0f}")
        print(f"   Random Forest     | R² = {rf_r2_scores.mean():.4f} ±{rf_r2_scores.std():.4f} | RMSE = ${rf_rmse:,.0f}")

        return results

    def train_final_model(self, X: pd.DataFrame, Y: pd.Series, model = None):
        if model is None:
            model = LinearRegression()
        model.fit(X, Y)
        return model

    def save_model(self, feature_columns: list, model, model_name: str = 'lr'):
        model_path = MODEL_DIR / f"{model_name}.joblib"
        cols_path = MODEL_DIR / f"{model_name}_columns.joblib"
        joblib.dump(model, model_path)
        joblib.dump(feature_columns, cols_path)

    def train(self):
        X, Y = self.prepare_features_target(self.load_data())

        self.evaluate_models(X, Y)

        final_model = self.train_final_model(X, Y)

        self.save_model(
            feature_columns= list(X.columns),
            model= final_model
        )

