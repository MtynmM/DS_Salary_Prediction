import pandas as pd
from pathlib import Path

class SalaryPredictor:
    """
    Main class to handle the entire ML pipeline:
    Loading -> Preprocessing -> Training -> Evaluation
    """
    def __init__(self, csv_filename):
        """
        Initializes the predictor with the dataset path.
        """
        #resolve=create complete path
        project_root = Path(__file__).resolve().parent.parent
        self.csv_path = project_root / "data" / csv_filename
        self.df = None

    def load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError('csv file not found...')
        self.df = pd.read_csv(self.csv_path)

    def preprocess_data(self):
        if self.df is None:
            raise ValueError('data not loaded! call load_csv() first...')
        
        #Ordinal Encoding: Experience Level
        exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
        self.df['experience_rank'] = self.df['experience_level'].map(exp_map)

        #Job Title Grouping(5first top)
        top_jobs = self.df['job_title'].value_counts().nlargest(5).tolist()
        self.df['job_group'] = self.df['job_title'].apply(lambda x: x if x in top_jobs else 'Other')

        #One-Hot Encoding
        categorical_col = ['company_size', 'employment_type', 'remote_ratio', 'job_group']
        self.df_encoded = pd.get_dummies(self.df, columns=categorical_col, drop_first=True)

        import numpy as np
        self.df_encoded['log_salary'] = np.log1p(self.df_encoded['salary_in_usd'])

        drop_cols = ['work_year', 'salary', 'salary_currency', 'salary_in_usd', 
                     'employee_residence', 'company_location', 'job_title', 'experience_level']

        self.final_df = self.df_encoded.drop(columns=[c for c in drop_cols if c in self.df_encoded.columns])


if __name__ == '__main__':
    app = SalaryPredictor('ds_salaries.csv')
    app.load_csv()
    app.preprocess_data()
    print(app.final_df.head())