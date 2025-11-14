import os
import sys
import logging
import pandas as pd
from typing import Dict
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from handling_outliers import CustomOutlierDetection
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy
from feature_engineering import NewFeatureEngineer
from handling_outliers import OutlierDetector
from scalling_and_encoding import RegressionPreprocessor,ClassificationPreprocessor,ClusteringPreprocessor
from data_splitter import SimpleTrainTestSplitStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data,get_ingestion,get_missing_values,get_outliers,get_feature_engineering,get_preprocessing,get_splitting
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags

def data_pipeline(
    data_path: str = 'data/raw/Viral_Social_Media_Trends.csv',
    )->Dict[str, np.ndarray]:

    get_data_config = get_data()
    ingestion_config = get_ingestion()
    missing_config = get_missing_values()
    outlier_config = get_outliers()
    feature_engineering_config = get_feature_engineering()
    reg_config = get_preprocessing(task="regression")
    cla_config = get_preprocessing(task="classification")
    clu_config = get_preprocessing(task="clustering")
    reg_split_config = get_splitting(task="regression")
    cla_split_config = get_splitting(task="classification")

    #mlflow intergrating
    mlflow_tracker = MLflowTracker()

    print("\n----------01 data ingestion-----------\n")

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)

    print("\n------------step 02: Handle missing values--------------\n")

    drop_handler = DropMissingValuesStrategy()
    df = drop_handler.handle(df)

    print("\n------------step 03: Handling Outliers--------------\n")

    outlier_handler = OutlierDetector(CustomOutlierDetection())
    df = outlier_handler.handle_outliers(df,selected_columns=outlier_config["selected_columns"],
                                        target_column=outlier_config["target_column"]
                                        )
    print(df.shape[0])

    print("\n------------step 04: Feature Engineering--------------\n")

    feature_handler = NewFeatureEngineer(means_file=feature_engineering_config['means_file'])
    feature_handler.fit(df)
    df = feature_handler.handle(df)

    print("\n------------step 05: Scalling and encoding--------------\n")

    reg_handler = RegressionPreprocessor(columns_to_drop=reg_config["columns_to_drop"],
                                         nominal_columns_reg= reg_config["nominal_columns"],
                                         numerical_columns_reg = reg_config["numerical_columns"],
                                         ordinal_columns_reg = reg_config["ordinal_columns"]
                                        )
    df_reg = reg_handler.handle(df)

    cla_handler = ClassificationPreprocessor(
                                    columns_to_keep=cla_config["columns_to_keep"],
                                    nominal_columns_cla = cla_config["nominal_columns"],
                                    numerical_columns_cla = cla_config ["numerical_columns"],
                                    target_column = cla_config["target_column"]
                                    )
    df_cla = cla_handler.handle(df)

    clu_handler = ClusteringPreprocessor(
                                    columns_to_keep=clu_config["columns_to_keep"],
                                    nominal_columns_cla = clu_config["nominal_columns"],
                                    numerical_columns_cla = clu_config ["numerical_columns"],
                                    )
    clu_handler.handle(df)

    print("\n------------step 06: Splitting data -------------\n")

    reg_splitter_handler = SimpleTrainTestSplitStrategy()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = reg_splitter_handler.split_data(df_reg,target_column=reg_split_config["target_column"])

    saving_path = reg_split_config["saving_path"]

    X_train_reg.to_csv(f"{saving_path}/X_train_reg.csv",index=False)
    X_test_reg.to_csv(f"{saving_path}/X_test_reg.csv",index=False)
    y_train_reg.to_csv(f"{saving_path}/y_train_reg.csv",index=False)
    y_test_reg.to_csv(f"{saving_path}/y_test_reg.csv",index=False)

    ########## Mlflow ##################
    regression_tags = create_mlflow_run_tags(
        'data_pipeline', 
        {'data_paths': data_path, 'model': 'regression'}
    )
    mlflow_tracker.start_run(run_name='data_pipeline_regression', tags=regression_tags)

    mlflow_tracker.log_data_pipeline_metrics({
                        'model': "Regression",
                        'total_rows': len(X_train_reg) + len(X_test_reg),
                        'train_rows': len(X_train_reg),
                        'test_rows': len(X_test_reg),
                        'num_features': X_train_reg.shape[1],
                        'missing_values': X_train_reg.isna().sum().sum(),
                        'outliers_removed': 270 
                    })
    mlflow_tracker.end_run()

    cla_splitter_handler = SimpleTrainTestSplitStrategy(stratify=True)

    X_train_cla, X_test_cla, y_train_cla, y_test_cla = cla_splitter_handler.split_data(df_cla,target_column=cla_split_config["target_column"])

    saving_path = cla_split_config["saving_path"]

    X_train_cla.to_csv(f"{saving_path}/X_train_cla.csv",index=False)
    X_test_cla.to_csv(f"{saving_path}/X_test_cla.csv",index=False)
    y_train_cla.to_csv(f"{saving_path}/y_train_cla.csv",index=False)
    y_test_cla.to_csv(f"{saving_path}/y_test_cla.csv",index=False)

     ########## Mlflow ##################
     
    classification_tags = create_mlflow_run_tags(
        'data_pipeline', 
        {'data_paths': data_path, 'model': 'classification'}
    )
    mlflow_tracker.start_run(run_name='data_pipeline_classification', tags=classification_tags)

    mlflow_tracker.log_data_pipeline_metrics({
                        'model': "classification",
                        'total_rows': len(X_train_cla) + len(X_test_cla),
                        'train_rows': len(X_train_cla),
                        'test_rows': len(X_test_cla),
                        'num_features': X_train_cla.shape[1],
                        'missing_values': X_train_cla.isna().sum().sum(),
                        'outliers_removed': 270 
                    })
    mlflow_tracker.end_run()

    # ---------- Pipeline Complete ----------
    print("\n🏁 ==============================================")
    print("     🎉 DATA PIPELINE EXECUTED SUCCESSFULLY 🎉")
    print("==================================================\n")


if __name__ == "__main__":
    data_pipeline()
