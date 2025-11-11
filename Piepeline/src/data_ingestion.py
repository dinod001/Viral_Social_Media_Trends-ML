import pandas as pd
from abc import ABC, abstractmethod
import logging

# Configure logging (optional but helpful for real pipelines)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataIngestor(ABC):
    """Abstract base class defining a data ingestion interface."""

    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """Ingest data from a source (file or link) and return a DataFrame."""
        pass


class DataIngestorCSV(DataIngestor):
    """Ingests data from a CSV file."""

    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        logging.info(f"Loading CSV file from: {file_path_or_link}")
        df = pd.read_csv(file_path_or_link)
        logging.info(f"CSV file loaded successfully. Shape: {df.shape}")
        return df


class DataIngestorExcel(DataIngestor):
    """Ingests data from an Excel file."""

    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        logging.info(f"Loading Excel file from: {file_path_or_link}")
        df = pd.read_excel(file_path_or_link)
        logging.info(f"Excel file loaded successfully. Shape: {df.shape}")
        return df
