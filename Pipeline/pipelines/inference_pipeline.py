import os
import sys
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Ensure project modules are importable
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

for path in [PROJECT_ROOT, SRC_DIR, UTILS_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from model_inference import load_inference_pipeline, InferenceInput  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "classification": [
        {
            "Platform": "YouTube",
            "Hashtag": "#viral",
            "Content_Type": "Video",
            "Region": "North America",
            "Views": 120000,
            "Likes": 5800,
            "Shares": 940,
            "Comments": 420,
        }
    ],
    "regression": [
        {
            "Platform": "Instagram",
            "Hashtag": "#fashion",
            "Content_Type": "Image",
            "Region": "Europe",
            "Views": 45000,
            "Likes": 2700,
            "Shares": 310,
            "Comments": 185,
            
        }
    ],
}


def _prepare_input(input_data: InferenceInput) -> InferenceInput:
    """
    Accepts a DataFrame, dict, list of dicts, or path to a CSV file and returns
    a form that can be passed to ModelInference.predict().
    """
    if input_data is None:
        raise ValueError("input_data is required for inference.")

    if isinstance(input_data, (pd.DataFrame, dict, list)):
        return input_data

    if isinstance(input_data, str):
        lower = input_data.lower()
        if lower in ("sample", "sample_classification"):
            return DEFAULT_SAMPLES["classification"]
        if lower == "sample_regression":
            return DEFAULT_SAMPLES["regression"]

        csv_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", input_data))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at '{csv_path}'.")
        logger.info("Loading inference data from %s", csv_path)
        return pd.read_csv(csv_path)

    raise TypeError("input_data must be DataFrame, dict, list of dicts, or CSV path.")


def inference_pipeline(
    task: str,
    input_data: InferenceInput,
    model_path: Optional[str] = None,
    return_proba: bool = True,
) -> Dict[str, Any]:
    """
    High-level helper that wraps ModelInference for quick predictions.

    Args:
        task: "classification" or "regression".
        input_data: DataFrame, dict, list of dicts, or CSV path with samples.
        model_path: Optional path to a specific model artifact.
        return_proba: If True (classification), also return class probabilities.

    Returns:
        Dictionary with predictions (and probabilities for classification).
    """
    prepared_data = _prepare_input(input_data)
    inference = load_inference_pipeline(task=task, model_path=model_path)
    logger.info("Running inference for %s task...", task)
    return inference.predict(prepared_data, return_proba=return_proba)


if __name__ == "__main__":
    logger.info("Example usage of inference_pipeline:")
    # Example with built-in samples
    try:
        sample_cls = inference_pipeline(task="classification", input_data="sample")
        logger.info("Sample classification inference: %s", sample_cls)

        sample_reg = inference_pipeline(task="regression", input_data="sample_regression")
        logger.info("Sample regression inference: %s", sample_reg)
    except Exception as exc:
        logger.warning("Sample inference failed: %s", exc)
