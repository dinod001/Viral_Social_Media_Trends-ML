import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP is not installed. Install using: pip install umap-learn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DimensionalityReduction")


class DimensionalityReducer:
    def __init__(self, method: str, n_components: int = 2):
        """
        Supported methods: 'pca', 'tsne', 'umap'
        """
        self.method = method.lower()
        self.n_components = n_components
        self.model = None

        self._initialize_method()

    def _initialize_method(self):
        if self.method == "pca":
            if self.n_components is None:
                self.model = PCA(n_components=2)
            else:
                self.model = PCA(n_components=self.n_components)

        elif self.method == "tsne":
            self.model = TSNE(n_components=self.n_components, random_state=42)

        elif self.method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP is not installed. Run: pip install umap-learn")
            self.model = umap.UMAP(n_components=self.n_components, random_state=42)

        else:
            raise ValueError("Invalid method. Choose from: 'pca', 'tsne', 'umap'")

        logger.info(f"Initialized dimensionality reduction method: {self.method.upper()}")

    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Apply dimensionality reduction.

        :param data: DataFrame or NumPy array
        :return: reduced-dimensional array
        """
        logger.info(f"Applying {self.method.upper()} with {self.n_components} components...")

        if isinstance(data, pd.DataFrame):
            data = data.values  # convert to numpy array

        reduced = self.model.fit_transform(data)

        logger.info(f"{self.method.upper()} reduction completed. Result shape: {reduced.shape}")

        return reduced

