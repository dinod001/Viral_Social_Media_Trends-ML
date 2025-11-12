import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.evaluation_results: Dict[str, Any] = {}
        logger.info(f"Initialized ModelEvaluator for model: {self.model_name}")

    def evaluate_classification(self, X_test, y_test):
        logger.info(f"Starting classification evaluation for {self.model_name}...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        logger.debug(f"Predictions completed. Sample predictions: {y_pred[:5]}")

        # Compute metrics
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
        recall = recall_score(y_test, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')

        self.evaluation_results = {
            "confusion_matrix": cm,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logger.info(f"[Classification] {self.model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                    f"Recall: {recall:.3f}, F1: {f1:.3f}")
        logger.debug(f"Confusion Matrix:\n{cm}")
        
        return self.evaluation_results

    def evaluate_regression(self, X_test, y_test):
        logger.info(f"Starting regression evaluation for {self.model_name}...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        logger.debug(f"Predictions completed. Sample predictions: {y_pred[:5]}")

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        self.evaluation_results = {
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }

        logger.info(f"[Regression] {self.model_name} - R²: {r2:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}")
        return self.evaluation_results
