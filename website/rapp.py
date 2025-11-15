from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pandas as pd

from feature_engineering import NewFeatureEngineer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.join(BASE_DIR, "preprocessor", "regression_preprocessor.joblib")
model_path = os.path.join(BASE_DIR, "model", "reg_socialMedia_analyzer.joblib")
scaler_path = os.path.join(BASE_DIR, "preprocessor", "shares_minmax_scaler.joblib")
preprocessor = joblib.load(preprocessor_path)
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    pred_value=0
    if request.method == 'POST':
        # Get all values from the form
        data = {
            "Platform": request.form.get('Platform'),
            "Hashtag": request.form.get('Hashtag'),
            "Content_Type": request.form.get('Content_Type'),
            "Region": request.form.get('Region'),
            "Views": int(request.form.get('Views', 0)),
            "Likes": int(request.form.get('Likes', 0)),
            "Comments": int(request.form.get('Comments', 0)),
            "Shares": int(request.form.get('Shares', 0)),
            "Engagement_Level": request.form.get('Engagement_Level')
        }

        df = pd.DataFrame([data])
        preprocessed_df = preprocess_input(df)
        pred_value = predict(preprocessed_df)
    return render_template('regression.html',pred_value=pred_value)

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering and preprocessing for regression or classification."""
    feature_handler = NewFeatureEngineer(means_file='platform_means.json')
    feature_handler.load_means()
    df = feature_handler.handle(df)

    columns_to_drop = ["Share_Rate","Total_Engagement","Engagement_Rate","Engagement_Level"]

    df_reg = df.drop(columns=columns_to_drop, errors='ignore')
    transformed_array = preprocessor.transform(df_reg)
    df_transformed = pd.DataFrame(
        transformed_array,
        columns=preprocessor.get_feature_names_out(),
        index=df_reg.index
    )
    # Rename columns
    rename_map = {col: col.split("__", 1)[-1] for col in df_transformed.columns}
    df_transformed = df_transformed.rename(columns=rename_map)
    df_transformed = df_transformed.drop(columns=["Shares"], errors="ignore")
    return df_transformed

def predict(preprocessed_df: pd.DataFrame):
    predictions = model.predict(preprocessed_df)
    predictions = int(scaler.inverse_transform(predictions.reshape(-1, 1)).flatten())
    return predictions

    # If GET request, just render the form
if __name__ == '__main__':
        app.run(debug=True) 