import os
import sys
import logging
import pandas as pd

# Add project src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_inference import ModelInference

# Configure main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InferencePipeline")


def create_sample_data() -> pd.DataFrame:
    data = {
        "Platform": "Instagram",
        "Hashtag": "#Challenge",
        "Content_Type": "Video",
        "Region": "UK",
        "Views": 45000,
        "Likes": 2700,
        "Comments": 185
    }

    df = pd.DataFrame([data])
    df['Shares'] = 0
    df['Engagement_Level'] = 'Low'
    return df


def run_inference(df: pd.DataFrame):
    """
    Run classification and regression inference, logging cleanly.
    """
   
    print("\n------------- CLASSIFICATION INFERENCE ---------------\n")

    try:
        clf_model = ModelInference(task='classification')
        pred_cls = clf_model.predict(df)
        logger.info(f"Classification Prediction: {pred_cls}")
    except Exception as e:
        logger.error(f"Classification inference failed: {e}")

    print("\n------------- REGRESSION INFERENCE ---------------\n")

    try:
        reg_model = ModelInference(task='regression')
        pred_reg = reg_model.predict(df)
        logger.info(f"Regression Prediction: {pred_reg}")
    except Exception as e:
        logger.error(f"Regression inference failed: {e}")


def inference_pipeline():
    logger.info("Starting inference pipeline...")
    df = create_sample_data()
    run_inference(df)
    logger.info("Inference pipeline completed.")


if __name__ == "__main__":
    inference_pipeline()
