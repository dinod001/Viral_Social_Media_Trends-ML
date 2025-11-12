import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, param_grid=None, cv=5, scoring='accuracy', n_jobs=-1):
        """
        param_grid: dict
            Hyperparameter grid for GridSearchCV.
        cv: int
            Number of cross-validation folds.
        scoring: str
            Scoring metric.
        n_jobs: int
            Number of parallel jobs (-1 uses all cores).
        """
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

        logger.info(f"Initialized ModelTrainer with scoring={scoring}, cv={cv}, n_jobs={n_jobs}")

    def train(self, X_train, Y_train, model):
        model_name = model.__class__.__name__
        logger.info(f"Starting training for model: {model_name}")

        if self.param_grid:
            logger.info(f"Hyperparameter tuning using GridSearchCV with {len(self.param_grid)} parameter sets...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
            logger.info("Fitting GridSearchCV...")
            grid_search.fit(X_train, Y_train)

            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

            logger.info(f"GridSearchCV completed for {model_name}")
            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Best Cross-Validation Score: {best_score:.4f}")
            return best_model, best_score
        else:
            logger.warning("No param_grid provided. Proceeding with default model parameters...")
            model.fit(X_train, Y_train)
            train_score = model.score(X_train, Y_train)
            logger.info(f"Model {model_name} trained with default parameters. Training score: {train_score:.4f}")
            return model, train_score

    def save_model(self, model, filePath):
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        joblib.dump(model, filePath)
        logger.info(f" Model saved successfully at: {filePath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise ValueError("Can't load. File not found.")
        logger.info(f"Loading model from: {filepath}")
        return joblib.load(filepath)
