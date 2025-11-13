import os
import sys
import pandas as pd
import joblib

# Set project paths
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "utils")):
    if path not in sys.path:
        sys.path.append(path)

from config import get_feature_engineering, get_preprocessing, get_model_training
from feature_engineering import NewFeatureEngineer

# Load configs
project_root = PROJECT_ROOT
feature_engineering_config = get_feature_engineering()
reg_config = get_preprocessing(task="regression")
cla_config = get_preprocessing(task="classification")

# Load preprocessors and encoders/scalers
base_path = reg_config["artifacts_path"]
cla_base_path = cla_config["encoder_path"]

reg_preprocessor = joblib.load(os.path.join(project_root, base_path, "regression_preprocessor.joblib"))
cla_preprocessor = joblib.load(os.path.join(project_root, base_path, "classification_preprocessor.joblib"))
reg_scaler = joblib.load(os.path.join(project_root, base_path, "shares_minmax_scaler.joblib"))
cla_encoder = joblib.load(os.path.join(project_root, cla_base_path, "classification_target_encoder.joblib"))


class ModelInference:
    def __init__(self, task: str):
        self.task = task
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from config path."""
        training_config = get_model_training(task=self.task)
        model_path = training_config.get("model_dir", "")
        full_path = os.path.join(project_root, model_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found at {full_path}")
        self.model = joblib.load(full_path)

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering and preprocessing for regression or classification."""
        feature_handler = NewFeatureEngineer(means_file=feature_engineering_config['means_file'])
        feature_handler.load_means()
        df = feature_handler.handle(df)

        if self.task == 'regression':
            df_reg = df.drop(columns=reg_config["columns_to_drop"], errors='ignore')
            transformed_array = reg_preprocessor.transform(df_reg)
            df_transformed = pd.DataFrame(
                transformed_array,
                columns=reg_preprocessor.get_feature_names_out(),
                index=df_reg.index
            )
            # Rename columns
            rename_map = {col: col.split("__", 1)[-1] for col in df_transformed.columns}
            df_transformed = df_transformed.rename(columns=rename_map)
            df_transformed = df_transformed.drop(columns=["Shares"], errors="ignore")
            return df_transformed

        elif self.task == 'classification':
            df_cla = df[cla_config['columns_to_keep']]
            transformed_array = cla_preprocessor.transform(df_cla)
            df_transformed = pd.DataFrame(
                transformed_array,
                columns=cla_preprocessor.get_feature_names_out(),
                index=df_cla.index
            )
            rename_map = {col: col.split("__", 1)[-1] for col in df_transformed.columns}
            df_transformed = df_transformed.rename(columns=rename_map)
            df_transformed = df_transformed.drop(columns=["Engagement_Level"], errors="ignore")
            return df_transformed

    def predict(self, df: pd.DataFrame):
        """Preprocess input, predict, and return original-scale outputs."""
        preprocessed_df = self.preprocess_input(df)
        predictions = self.model.predict(preprocessed_df)

        if self.task == 'regression':
            # Convert back to original Shares values
            predictions = int(reg_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten())
        elif self.task == 'classification':
            # Convert back to original labels
            predictions = cla_encoder.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions
