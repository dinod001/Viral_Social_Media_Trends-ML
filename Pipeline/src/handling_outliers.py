import pandas as pd
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OutlierDetectionStrategy(ABC):
    """Abstract base class for outlier detection strategies."""

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list, target_column: str) -> pd.DataFrame:
        """Return a DataFrame with outliers removed based on the implemented strategy."""
        pass


class CustomOutlierDetection(OutlierDetectionStrategy):
    """Detects logical outliers based on a target reference column (e.g., Views)."""

    def detect_outliers(self, df: pd.DataFrame, columns: list, target_column: str) -> pd.DataFrame:
        logging.info(f"Starting outlier detection using CustomOutlierDetection strategy...")
        initial_rows = len(df)
        df_filtered = df.copy()
        df_filtered = df_filtered.reset_index(drop=True)

        for col in columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered[col] <= df_filtered[target_column]]
            removed = before - len(df_filtered)
            logging.info(f"Filtered {removed} rows where {col} > {target_column}")

        total_removed = initial_rows - len(df_filtered)
        logging.info(f"Outlier removal complete. Total rows removed: {total_removed}")
        logging.info(f"Data shape before: {initial_rows}, after: {len(df_filtered)}")

        return df_filtered


class OutlierDetector:
    """High-level class that applies an outlier detection strategy."""

    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, selected_columns: list, target_column: str) -> pd.DataFrame:
        logging.info("Detecting outliers...")
        return self.strategy.detect_outliers(df, selected_columns, target_column)

    def handle_outliers(self, df: pd.DataFrame, selected_columns: list, target_column: str) -> pd.DataFrame:
        """Apply the outlier detection and return a cleaned DataFrame."""
        logging.info("Handling outliers with selected strategy...")
        df_cleaned = self.detect_outliers(df, selected_columns, target_column)
        logging.info("Outlier handling completed successfully.")
        return df_cleaned
