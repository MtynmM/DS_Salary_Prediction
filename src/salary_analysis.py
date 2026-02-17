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

if __name__ == '__main__':
    app = SalaryPredictor('ds_salaries.csv')
    app.load_csv()