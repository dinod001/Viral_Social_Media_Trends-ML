import os
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Preprocessor(ABC):
    """Abstract base class for preprocessing."""

    @abstractmethod
    def handle(self, df: pd.DataFrame):
        pass


class RegressionPreprocessor(Preprocessor):
    """Preprocess dataset for regression."""

    def __init__(self, columns_to_drop=None,
                 nominal_columns_reg=None,
                 numerical_columns_reg=None,
                 ordinal_columns_reg=None,
                 save_path='data/processed/',
                 artifacts_path='artifacts/preprocessor/'):
        self.columns_to_drop = columns_to_drop or []
        self.nominal_columns_reg = nominal_columns_reg or []
        self.numerical_columns_reg = numerical_columns_reg or []
        self.ordinal_columns_reg = ordinal_columns_reg or []
        self.save_path = save_path
        self.artifacts_path = artifacts_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)

    def handle(self, df: pd.DataFrame):
        logging.info("Starting regression preprocessing...")
        df_reg = df.drop(columns=self.columns_to_drop, errors='ignore')

        numerical_cols = self.numerical_columns_reg
        nominal_cols = self.nominal_columns_reg
        ordinal_cols = self.ordinal_columns_reg

        transformers = []

        # ---------------------------
        # 1️⃣ Scale numerical columns
        # ---------------------------
        if numerical_cols:
            scaler = MinMaxScaler()
            transformers.append(('num', Pipeline([('scaler', scaler)]), numerical_cols))

            # Save only the Shares scaler
            if 'Shares' in numerical_cols:
                shares_scaler = MinMaxScaler()
                shares_scaler.fit(df_reg[['Shares']].values)  # fit only on Shares
                joblib.dump(
                    shares_scaler,
                    os.path.join(self.artifacts_path, 'shares_minmax_scaler.joblib')
                )
                logging.info("Saved Shares MinMaxScaler separately.")

        # ---------------------------
        # 2️⃣ Encode nominal columns
        # ---------------------------
        if nominal_cols:
            transformers.append(
                ('nom', Pipeline([
                    ('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
                ]), nominal_cols)
            )

        # ---------------------------
        # 3️⃣ Encode ordinal columns
        # ---------------------------
        if ordinal_cols:
            transformers.append(
                ('ord', Pipeline([('encoder', OrdinalEncoder())]), ordinal_cols)
            )

        # ---------------------------
        # 4️⃣ Apply ColumnTransformer
        # ---------------------------
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        transformed = preprocessor.fit_transform(df_reg)

        # ---------------------------
        # 5️⃣ Reconstruct DataFrame
        # ---------------------------
        all_features = numerical_cols.copy()
        if nominal_cols:
            nom_names = preprocessor.named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(nominal_cols)
            all_features += list(nom_names)
        all_features += ordinal_cols

        df_transformed = pd.DataFrame(transformed, columns=all_features, index=df_reg.index)

        # ---------------------------
        # 6️⃣ Save processed data & preprocessor
        # ---------------------------
        csv_path = os.path.join(self.save_path, 'regression_scaled_encoded.csv')
        df_transformed.to_csv(csv_path, index=False)

        joblib.dump(preprocessor, os.path.join(self.artifacts_path, 'regression_preprocessor.joblib'))

        logging.info(f"Regression preprocessing done. Saved CSV: {csv_path}, shape: {df_transformed.shape}")
        return df_transformed


class ClassificationPreprocessor(Preprocessor):
    """Preprocess dataset for classification."""

    def __init__(self, columns_to_keep=None,
                 nominal_columns_cla=None,
                 numerical_columns_cla=None,
                 target_column=None,
                 save_path='data/processed/',
                 artifacts_path='artifacts/preprocessor/',
                 encoder_path='artifacts/encoder/'):
        self.columns_to_keep = columns_to_keep or []
        self.nominal_columns_cla = nominal_columns_cla or []
        self.numerical_columns_cla = numerical_columns_cla or []
        self.target_column = target_column
        self.save_path = save_path
        self.artifacts_path = artifacts_path
        self.encoder_path = encoder_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(self.encoder_path, exist_ok=True)

    def handle(self, df: pd.DataFrame):
        logging.info("Starting classification preprocessing...")

        # Keep only selected columns
        df_cla = df[self.columns_to_keep]

        numerical_cols = self.numerical_columns_cla
        nominal_cols = self.nominal_columns_cla

        transformers = []
        if numerical_cols:
            transformers.append(('num', Pipeline([('scaler', MinMaxScaler())]), numerical_cols))
        if nominal_cols:
           transformers.append(('nom', Pipeline([('encoder', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))]), nominal_cols))

        preprocessor = ColumnTransformer(transformers, remainder='drop')

        transformed = preprocessor.fit_transform(df_cla)

        all_features = numerical_cols.copy()
        if nominal_cols:
            nom_names = preprocessor.named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(nominal_cols)
            all_features += list(nom_names)

        df_transformed = pd.DataFrame(transformed, columns=all_features, index=df_cla.index)

        # Encode target if present
        if self.target_column in df_cla.columns:
            target_encoder = OrdinalEncoder()
            df_transformed[self.target_column] = target_encoder.fit_transform(df_cla[[self.target_column]])
            joblib.dump(target_encoder, os.path.join(self.encoder_path, 'classification_target_encoder.joblib'))

        # Save results
        csv_path = os.path.join(self.save_path, 'classification_scaled_encoded.csv')
        df_transformed.to_csv(csv_path, index=False)
        joblib.dump(preprocessor, os.path.join(self.artifacts_path, 'classification_preprocessor.joblib'))

        logging.info(f"Classification preprocessing done. Saved CSV: {csv_path}, shape: {df_transformed.shape}")
        return df_transformed

class ClusteringPreprocessor(Preprocessor):
    """Preprocess dataset for clustering."""

    def __init__(self, columns_to_keep=None,
                 nominal_columns_cla=None,
                 numerical_columns_cla=None,
                 artifacts_path='artifacts/preprocessor/',
                 save_path='data/processed/'
                ):
        
        self.columns_to_keep = columns_to_keep or []
        self.nominal_columns_cla = nominal_columns_cla or []
        self.numerical_columns_cla = numerical_columns_cla or []
        self.artifacts_path = artifacts_path
        self.save_path = save_path

    def handle(self, df: pd.DataFrame):
        logging.info("Starting clustering preprocessing...")

        # Keep only selected columns
        df = df[self.columns_to_keep]

        numerical_columns = self.numerical_columns_cla
        nominal_columns = self.nominal_columns_cla
        ordinal_columns = []

        numerical_transformer = Pipeline(
            steps=[('scaler', StandardScaler())]   # <-- Standardization (Z-score)
        )

        nominal_transformer = Pipeline(
            steps=[('encoder', OneHotEncoder(sparse_output=False))]
        )

        ordinal_transformer = Pipeline(
            steps=[('encoder', OrdinalEncoder())]
        )

        # === Combine transformations ===
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('nom', nominal_transformer, nominal_columns),
                ('ord', ordinal_transformer, ordinal_columns)
            ],
            remainder='drop'
        )

        # === Apply transformations ===
        transformed_array = preprocessor.fit_transform(df)

        # Get encoded column names for nominal features
        nominal_feature_names = preprocessor.named_transformers_['nom'] \
            .named_steps['encoder'].get_feature_names_out(nominal_columns)

        # Combine all feature names
        all_feature_names = numerical_columns + list(nominal_feature_names) + ordinal_columns

        # Create new DataFrame
        df_transformed = pd.DataFrame(transformed_array, columns=all_feature_names)
        df_transformed.index = df.index  # Optional, keep same index

        #saving
        csv_path = os.path.join(self.save_path, 'clustering_scaled_encoded.csv')
        df_transformed.to_csv(csv_path, index=False)
        joblib.dump(preprocessor, os.path.join(self.artifacts_path, 'clustering_preprocessor.joblib'))
        logging.info(f"Clustering preprocessing done. Saved CSV: {csv_path}, shape: {df_transformed.shape}")