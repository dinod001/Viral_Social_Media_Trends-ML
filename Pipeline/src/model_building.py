import joblib, os
import logging
from xgboost import XGBClassifier, XGBRegressor
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModelBuilder(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        logger.info(f"Initialized {self.model_name} builder with params: {self.model_params}")

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model '{self.model_name}' saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError(f"Can't load. File not found: {filepath}")
        self.model = joblib.load(filepath)
        logger.info(f"Loaded model '{self.model_name}' from {filepath}")
        return self.model


# Classification
class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'criterion': 'entropy',
            'max_depth': 10,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)

    def build_model(self):
        logger.info("Building RandomForestClassifier model...")
        self.model = RandomForestClassifier(**self.model_params)
        logger.info(f"RandomForestClassifier built with params: {self.model_params}")
        return self.model


class XGboostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 10,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('XGboost', **default_params)

    def build_model(self):
        logger.info("Building XGBClassifier model...")
        self.model = XGBClassifier(**self.model_params)
        logger.info(f"XGBClassifier built with params: {self.model_params}")
        return self.model


# Regression
class RandomForestRegressorModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 8,
            'n_estimators': 100,
            'min_samples_split': 2,
            'min_samples_leaf': 5,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('RandomForestRegressor', **default_params)

    def build_model(self):
        logger.info("Building RandomForestRegressor model...")
        self.model = RandomForestRegressor(**self.model_params)
        logger.info(f"RandomForestRegressor built with params: {self.model_params}")
        return self.model


class XGBRegressorModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8
        }
        default_params.update(kwargs)
        super().__init__('XGBRegressor', **default_params)

    def build_model(self):
        logger.info("Building XGBRegressor model...")
        self.model = XGBRegressor(**self.model_params)
        logger.info(f"XGBRegressor built with params: {self.model_params}")
        return self.model
