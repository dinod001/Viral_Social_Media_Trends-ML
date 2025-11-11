import pandas as pd
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FeatureEngineer(ABC):
    """Abstract base class for feature engineering operations."""

    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by engineering new features."""
        pass


class NewFeatureEngineer(FeatureEngineer):
    """Performs new feature engineering transformations on the dataset."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement and interaction rate features."""

        # Ensure required columns exist
        required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'Platform']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        logging.info("Starting feature engineering...")

        # Avoid division by zero
        df = df.copy()
        df['Views'] = df['Views'].replace(0, pd.NA)
        df['Likes'] = df['Likes'].replace(0, pd.NA)

        # Engagement rates
        df['Like_Rate'] = df['Likes'] / df['Views']
        df['Share_Rate'] = df['Shares'] / df['Views']
        df['Comment_Rate'] = df['Comments'] / df['Views']
        df['Engagement_Rate'] = (df['Likes'] + df['Shares'] + df['Comments']) / df['Views']

        # Interaction ratios
        df['Like_to_Comment_Ratio'] = df['Likes'] / (df['Comments'] + 1)
        df['Share_to_Like_Ratio'] = df['Shares'] / (df['Likes'] + 1)

        # Platform-normalized metrics
        df['Views_Norm'] = df.groupby('Platform')['Views'].transform(lambda x: x / x.mean())

        logging.info("Feature engineering completed successfully.")
        return df
