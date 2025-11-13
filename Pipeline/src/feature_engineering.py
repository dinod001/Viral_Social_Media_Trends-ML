import pandas as pd
from abc import ABC, abstractmethod
import logging
import json
import os

# ------------------------------------------------------
# Configure logging
# ------------------------------------------------------
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

    def __init__(self, means_file: str = "platform_means.json"):
        """
        Args:
            means_file (str): Path to JSON file for storing platform view means.
        """
        self.means_file = means_file
        self.platform_view_means = None

    def fit(self, df: pd.DataFrame):
        """Compute and store platform mean views."""
        if "Platform" not in df.columns or "Views" not in df.columns:
            raise ValueError("DataFrame must contain 'Platform' and 'Views' columns to fit means.")

        self.platform_view_means = df.groupby("Platform")["Views"].mean().to_dict()
        logging.info(f"Computed platform means: {self.platform_view_means}")

        # Save the dictionary to a JSON file
        with open(self.means_file, "w") as f:
            json.dump(self.platform_view_means, f, indent=4)
        logging.info(f"Platform view means saved to '{self.means_file}'")

    def load_means(self):
        """Load previously saved platform mean views from file."""
        if not os.path.exists(self.means_file):
            raise FileNotFoundError(f"Means file '{self.means_file}' not found.")
        with open(self.means_file, "r") as f:
            self.platform_view_means = json.load(f)
        logging.info(f"Loaded platform means from '{self.means_file}'")


    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement, interaction, and normalized view features."""
        required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'Platform']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        if self.platform_view_means is None:
            raise RuntimeError("Platform means not set. Call `fit()` during training or `load_means()` before inference.")

        logging.info("Starting feature engineering...")
        df = df.copy()

        # Engagement rates
        df['Like_Rate'] = (df['Likes'] / df['Views'])
        df['Share_Rate'] = (df['Shares'] / df['Views'])
        df['Comment_Rate'] = (df['Comments'] / df['Views'])
        
        df['Total_Engagement'] = df[['Likes', 'Shares', 'Comments']].sum(axis=1)
        df['Engagement_Rate'] = (df['Total_Engagement'] / df['Views'])

        df['Total_Engagement_wo_Shares'] = df['Likes'] + df['Comments']
        df['Engagement_Rate_wo_Shares'] = (df['Likes'] + df['Comments']) / df['Views']

        # Platform-normalized metrics using saved means
        df['Views_Norm'] = df.apply(
            lambda row: row['Views'] / self.platform_view_means.get(row['Platform'], 1),
            axis=1
        )

        logging.info("Feature engineering completed successfully.")
        return df
