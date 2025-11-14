import joblib, os
import logging
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,DBSCAN

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

class KmeanClustering(BaseModelBuilder):
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.model = self.build_model()
    
    def build_model(self):
        logger.info("Building Kmeans model using silhouette score")
        K = range(2,11)
        silhouette_scores = {}
        for k in K:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(self.df)
            score = silhouette_score(self.df, labels)
            silhouette_scores[k] = score
        
        best_k = max(silhouette_scores,key=silhouette_scores.get)
        self.model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        logger.info(" ✅  Building Kmeans model Successfully ")
        return self.model

class DBSCANClustering(BaseModelBuilder):
    def __init__(self, df: pd.DataFrame, eps_values=None, min_samples=5):
        self.df = df
        self.min_samples = min_samples
        self.eps_values = eps_values or [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
        self.model = self.build_model()

    def build_model(self):
        logger.info("Building DBSCAN model using silhouette score to select best eps...")

        best_eps = None
        best_score = -1

        for eps in self.eps_values:
            model = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = model.fit_predict(self.df)

            # DBSCAN can produce noise (-1), silhouette fails if 1 cluster
            if len(set(labels)) <= 1:
                continue

            score = silhouette_score(self.df, labels)
            logger.info(f"eps={eps} → Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_eps = eps

        logger.info(f"Best eps selected: {best_eps} (Silhouette = {best_score:.4f})")
        self.model = DBSCAN(eps=best_eps, min_samples=self.min_samples)
        self.model.fit(self.df)
        logger.info(" ✅  Building Dbscan model Successfully ")
        return self.model

