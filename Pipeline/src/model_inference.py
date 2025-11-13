import os
import sys
from typing import Any, Dict

import joblib
import pandas as pd

# Make sure shared modules are importable
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils")):
    if path not in sys.path:
        sys.path.append(path)

from config import get_feature_engineering, get_preprocessing  # noqa: E402
from feature_engineering import NewFeatureEngineer

project_root = PROJECT_ROOT

feature_engineering_config = get_feature_engineering()
reg_config = get_preprocessing(task="regression")

base_path = reg_config["artifacts_path"]
joblib_path = os.path.join(project_root, base_path, "regression_preprocessor.joblib")
reg_preprocessor = joblib.load(joblib_path)

class ModelInference:
    def __init__(self, task: str):
        self.task = task
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # Feature engineering
        feature_handler = NewFeatureEngineer(means_file=feature_engineering_config['means_file'])
        feature_handler.load_means()
        df = feature_handler.handle(df)

        if self.task == 'regression':
            df_reg = df.drop(columns=reg_config["columns_to_drop"], errors='ignore')
            
            # Transform using saved preprocessor
            transformed_array = reg_preprocessor.transform(df_reg)
            
            # Convert back to DataFrame with proper feature names
            df_transformed = pd.DataFrame(
                transformed_array,
                columns=reg_preprocessor.get_feature_names_out(),
                index=df_reg.index
            )
            return df_transformed

data= {
        "Platform": "Instagram",
        "Hashtag": "#Challenge",
        "Content_Type": "Video",
        "Region": "UK",
        "Views": 45000,
        "Likes": 2700,
        "Comments": 185,
        "Engagement_Level":"High"
        } 

df = pd.DataFrame([data])

# Create inference object
model = ModelInference(task='regression')

# Preprocess the sample data
preprocessed_df = model.preprocess_input(df)
print(preprocessed_df)
    