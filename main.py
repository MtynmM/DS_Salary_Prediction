from pathlib import Path
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent

from src.data_pipeline.data_preparator import DataPreparator
from src.models.trainer import ModelTrainer
from src.api.predictor import Predictor
from src.api.main_api import create_app

db_path = ROOT_DIR / 'data' / 'db' / 'salary_data.db'
model_path = ROOT_DIR / 'data' / 'artifact' / 'lr.joblib'

def main():
    """Main project management"""
    
    if not model_path.exists():
        if not db_path.exists():
            print(" Database not found. Running ETL...")
            preparator = DataPreparator('ds_salaries.csv')
            preparator.etl()
            print(" ETL completed successfully ")
        print("training model..")
        trainer = ModelTrainer()
        trainer.train()
        print(" The model was trained and saved ")

    print(" Launch the API...")
    predictor = Predictor()
    app = create_app(predictor)
    #host="127.0.0.1"->local host
    #(Writing host and port is not mandatory(you can write uvicorn.run(app)))
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()