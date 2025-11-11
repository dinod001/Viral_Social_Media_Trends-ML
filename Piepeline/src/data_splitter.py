import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Abstract base class for all data splitting strategies."""

    @abstractmethod
    def split_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple train-test split strategy with optional stratification."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42, apply_smote: bool = False,stratify: bool = False):
        self.test_size = test_size
        self.random_state = random_state
        self.use_stratify = stratify  # placeholder for future use

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the DataFrame into train and test sets."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        stratify = y if self.use_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify= stratify
        )

        logger.info(
            f"Train-test split completed: "
            f"Train shape = {X_train.shape}, Test shape = {X_test.shape}, "
            f"Target = '{target_column}'"
        )

        return X_train, X_test, y_train, y_test
