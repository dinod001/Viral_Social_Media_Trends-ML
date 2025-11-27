# Viral Social Media Trends Finder

## Project Overview
This project aims to analyze social media trends and predict viral content using machine learning. It includes a comprehensive pipeline for data analysis, model training, and a web application for user interaction.

## Directory Structure

The project is organized into the following main directories:

- **EDA**: Contains Jupyter notebooks for Exploratory Data Analysis.
    - `01_handling_missing_values.ipynb`: Strategies for handling missing data.
    - `02_handling_outliers.ipynb`: Detection and treatment of outliers.
    - `03_feature_engineering.ipynb`: Creating new features from existing data.
    - `04_Data_visualization.ipynb`: Visualizing data distributions and relationships.
    - `05_encoding_and_scalling.ipynb`: Encoding categorical variables and scaling numerical features.
    - `06_encoding_and_standarlization.ipynb`: Standardization techniques.

- **Model Training**: Scripts and notebooks for training various machine learning models.
    - `Classification`: Models for classifying trends (e.g., Viral/Not Viral).
    - `Clustering`: Unsupervised learning to group similar trends.
    - `Regression`: Predicting continuous metrics like engagement scores.

- **Pipeline**: The core MLOps pipeline.
    - `src`: Source code for data processing and model definitions.
    - `pipelines`: Orchestration of data and training workflows.
    - `utils`: Utility functions.
    - `config.yaml`: Configuration settings.
    - `requirements.txt`: Dependencies for the pipeline.

- **website**: A Flask-based web application to demonstrate the models.
    - `classification_app.py`: Web app for classification models.
    - `regression_app.py`: Web app for regression models.
    - `templates`: HTML templates for the web interface.
    - `requirements.txt`: Dependencies for the web app.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Viral_Social_Media_Trends-Finder
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    For the pipeline:
    ```bash
    pip install -r Pipeline/requirements.txt
    ```
    For the website:
    ```bash
    pip install -r website/requirements.txt
    ```

## Usage

### Exploratory Data Analysis
Navigate to the `EDA` directory and run the Jupyter notebooks to understand the data processing steps.

### Running the Pipeline
Navigate to the `Pipeline` directory. You can configure the run in `config.yaml`.
(Add specific commands here if available, e.g., `python src/main.py`)

### Running the Web Application
Navigate to the `website` directory and run the Flask app:

For Classification:
```bash
python classification_app.py
```

For Regression:
```bash
python regression_app.py
```

The application will typically run on `http://localhost:5000`.

## Features
- **Trend Analysis**: Identify emerging trends from social media data.
- **Predictive Modeling**: Classify viral content and predict engagement.
- **Interactive Web Interface**: User-friendly dashboard to interact with the models.
- **MLOps Integration**: Structured pipeline for reproducible experiments (using MLflow/ZenML concepts).

## License
[License Name]
