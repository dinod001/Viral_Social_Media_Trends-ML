import pandas as pd
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MissingValueHandlingStrategy(ABC):
    """Abstract base class for missing value handling strategies."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the provided DataFrame."""
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """Concrete strategy that drops rows with missing or duplicate values."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows containing any nulls or duplicates."""

        original_len = len(df)

        # Drop rows with any null values
        df_cleaned = df.dropna()

        # Drop duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()

        # Log how many rows were dropped
        n_dropped = original_len - len(df_cleaned)
        logging.info(f"{n_dropped} rows dropped (missing or duplicate values).")

        return df_cleaned
