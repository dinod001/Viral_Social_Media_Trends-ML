import pandas as pd
import logging
from abc import ABC,abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        #dropiing any null values
        df_cleaned = df.dropna()

        # Drop duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f"{n_dropped} has been dropped")
        return df_cleaned
