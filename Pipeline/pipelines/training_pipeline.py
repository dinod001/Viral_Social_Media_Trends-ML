import os
import sys
import logging
import pandas as pd
from data_pipeline import data_pipeline
from typing import Dict, Any, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from sklearn.model_selection import KFold, StratifiedKFold

from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import (
    XGboostModelBuilder,
    RandomForestModelBuilder,
    RandomForestRegressorModelBuilder,
    XGBRegressorModelBuilder
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import (
    get_model_building,
    get_model_training,
    get_model_evaluation,
    get_splitting
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configurations
logger.info("Loading configuration files...")
reg_split_config = get_splitting(task="regression")
cla_split_config = get_splitting(task="classification")
saving_path_reg = reg_split_config["saving_path"]
saving_path_cla = cla_split_config["saving_path"]
reg_training_config = get_model_training(task="regression")
cla_training_config = get_model_training(task="classification")
logger.info("Configuration files loaded successfully.")


def training_pipeline(
        model_name: str = 'classification',
        model_params: Optional[Dict[str, Any]] = None,
        cv=5
    ):

    logger.info(f"Starting training pipeline for model: {model_name}")

    # Check if data artifacts exist
    if ((len(os.listdir(saving_path_cla)) == 0) | (len(os.listdir(saving_path_reg)) == 0)):
        logger.warning("Data artifacts not found. Running data pipeline to regenerate them...")
        data_pipeline()
    else:
        logger.info("Data artifacts found. Loading preprocessed data...")

    # ---------------- Regression ---------------- #
    if model_name == 'regression':
        logger.info("Loading regression data splits...")
        X_train = pd.read_csv(f"{saving_path_reg}/X_train_reg.csv")
        X_test = pd.read_csv(f"{saving_path_reg}/X_test_reg.csv")
        Y_train = pd.read_csv(f"{saving_path_reg}/Y_train_reg.csv")
        Y_test = pd.read_csv(f"{saving_path_reg}/Y_test_reg.csv")

        logger.info(f"Training regression model using XGBRegressor with parameters: {model_params}")
        model_path = reg_training_config["model_dir"]
        model_builder = XGBRegressorModelBuilder()
        model = model_builder.build_model()

        trainer = ModelTrainer(param_grid=model_params, cv=cv)
        model, train_score = trainer.train(
            model=model,
            X_train=X_train,
            Y_train=Y_train
        )

        trainer.save_model(model, model_path)
        logger.info(f"Model saved successfully at: {model_path}")

        evaluator = ModelEvaluator(model, "XGBRegressor")
        evaluation_results = evaluator.evaluate_regression(X_test, Y_test)

        logger.info(f"Regression Evaluation Results: {evaluation_results}")
        print(evaluation_results)

    # ---------------- Classification ---------------- #
    elif model_name == 'classification':
        logger.info("Loading classification data splits...")
        X_train = pd.read_csv(f"{saving_path_cla}/X_train_cla.csv")
        X_test = pd.read_csv(f"{saving_path_cla}/X_test_cla.csv")
        Y_train = pd.read_csv(f"{saving_path_cla}/Y_train_cla.csv")
        Y_test = pd.read_csv(f"{saving_path_cla}/Y_test_cla.csv")

        logger.info(f"Training classification model using XGBoost with parameters: {model_params}")
        model_path = cla_training_config["model_dir"]
        model_builder = XGboostModelBuilder()
        model = model_builder.build_model()

        trainer = ModelTrainer(param_grid=model_params)
        model, train_score = trainer.train(
            model=model,
            X_train=X_train,
            Y_train=Y_train
        )

        trainer.save_model(model, model_path)
        logger.info(f"Model saved successfully at: {model_path}")

        evaluator = ModelEvaluator(model, "XGboost")
        evaluation_results = evaluator.evaluate_classification(X_test, Y_test)

        evaluation_results_cp = evaluation_results.copy()
        del evaluation_results_cp['confusion_matrix']

        logger.info(f"Classification Evaluation Results: {evaluation_results_cp}")
        print(evaluation_results)

    logger.info(f"✅ Training pipeline completed successfully for {model_name} model.")


if __name__ == "__main__":
    logger.info("Initializing cross-validation strategies...")

    kf_config = reg_training_config["kfold"]
    kf = KFold(**kf_config)

    skf_config = cla_training_config["stratified_kfold"]
    skf = StratifiedKFold(**skf_config)
    
    logger.info("Cross-validation strategies initialized successfully.")

    logger.info("🚀 Starting model training process...")
    training_pipeline(
        model_name="classification",
        model_params=reg_training_config["param_grids"]["XGBoost"],
        cv=skf
    )
