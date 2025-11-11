import pandas as pd
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class OutlierDetectionStrategy(ABC):
    """Abstract base class for outlier detection strategies."""

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Return a boolean DataFrame marking outliers (True = outlier)."""
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """Detects outliers using the Interquartile Range (IQR) method."""

    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        outliers = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            # Ensure numeric dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            logging.info(f"Column '{col}': IQR={IQR:.2f}, "
                         f"Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, "
                         f"Outliers={outliers[col].sum()}")

        return outliers


class OutlierDetector:
    """High-level class that applies an outlier detection strategy."""

    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
        return self.strategy.detect_outliers(df, selected_columns)

    def handle_outliers(self, df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
        """Remove rows that contain outliers in two or more of the selected columns."""
        outliers = self.detect_outliers(df, selected_columns)
        outlier_counts = outliers.sum(axis=1)
        rows_to_remove = outlier_counts >= 2
        n_removed = rows_to_remove.sum()

        logging.info(f"{n_removed} rows removed due to multiple outliers.")
        return df[~rows_to_remove].reset_index(drop=True)
