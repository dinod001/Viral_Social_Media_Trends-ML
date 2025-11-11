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
from handling_outliers import IQROutlierDetection
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy
from feature_engineering import NewFeatureEngineer
from handling_outliers import OutlierDetector
from scalling_and_encoding import RegressionPreprocessor,ClassificationPreprocessor
from data_splitter import SimpleTrainTestSplitStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data,get_ingestion,get_missing_values,get_outliers,get_feature_engineering,get_preprocessing,get_splitting

def data_pipeline(
    data_path: str = 'data/raw/Viral_Social_Media_Trends.csv',
    )->Dict[str, np.ndarray]:

    get_data_config = get_data()
    ingestion_config = get_ingestion()
    missing_config = get_missing_values()
    outlier_config = get_outliers()
    feature_engineering_config = get_feature_engineering()
    #preprocessing_config = get_preprocessing()
    #splitting_config = get_splitting()

    print("\n----------01 data ingestion-----------\n")

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)

    print("\n------------step 02: Handle missing values--------------\n")

    drop_handler = DropMissingValuesStrategy()
    df = drop_handler.handle(df)

    print("\n------------step 03: Feature Engineering--------------\n")

    feature_handler = NewFeatureEngineer(means_file=feature_engineering_config['means_file'])
    feature_handler.fit(df)
    df = feature_handler.handle(df)

    print("\n------------step 04: Handling Outliers--------------\n")

    outlier_handler = OutlierDetector(IQROutlierDetection())
    df = outlier_handler.handle_outliers(df,selected_columns=outlier_config["selected_columns"])
    print(df.shape[0])

if __name__ == "__main__":
    data_pipeline()
