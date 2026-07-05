import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Feature Loading Function ---
def load_test_features(file_path):
    """Loads testing features and extracts input matrices and true labels."""
    try:
        logger.info(f"Loading test features from {file_path}...")
        test_data = pd.read_csv(file_path)
        if test_data.empty:
            raise ValueError(f"The feature dataset at {file_path} is empty.")
        
        # FIX: Slices features and final target columns correctly using standard integer slicing
        X_test = test_data.iloc[:, 0:-1].values
        y_test_raw = test_data.iloc[:, -1].values
        
        logger.info(f"Loaded test feature matrix shape: {X_test.shape}")
        return X_test, y_test_raw
    except FileNotFoundError:
        logger.error(f"Test feature file not found at: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error while loading test feature files: {e}")
        raise

# --- Model Artifacts Loading Function ---
def load_pipeline_artifacts(models_dir="models"):
    """Loads the trained model file and the fitted label encoder."""
    try:
        logger.info(f"Loading pipeline artifacts from directory: {models_dir}")
        model_path = os.path.join(models_dir, "model.pkl")
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
        logger.info("Successfully unpickled trained XGBoost model.")
        
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Successfully unpickled fitted label encoder.")
        
        return xgb_model, label_encoder
    except FileNotFoundError as e:
        logger.error(f"Required artifact file missing in {models_dir}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to unpickle model tracking files: {e}")
        raise

# --- Model Evaluation Function ---
def evaluate_model_performance(xgb_model, label_encoder, X_test, y_test_raw):
    """Generates evaluation metrics and structural report matrices."""
    try:
        logger.info("Transforming target test labels and running predictions...")
        # Match test labels to training encoding rules
        y_test_encoded = label_encoder.transform(y_test_raw)
        
        # Generate model predictions
        y_pred = xgb_model.predict(X_test)
        
        logger.info("Calculating operational pipeline evaluation metrics...")
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(
            y_test_encoded, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test_encoded, y_pred, average="weighted", zero_division=0
        )
        classification_rep = classification_report(
            y_test_encoded, y_pred, zero_division=0
        )
        
        # Output readable metrics report to console log stream
        logger.info(f"Model Evaluation Metrics:\n"
                    f" -> Accuracy : {accuracy:.4f}\n"
                    f" -> Precision: {precision:.4f}\n"
                    f" -> Recall   : {recall:.4f}")
        
        print(f"\nClassification Report:\n{classification_rep}")
        
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
        }
        return metrics_dict
    except ValueError as e:
        logger.error(f"Label mismatch error during label transformation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during calculations inside model evaluator: {e}")
        raise

# --- Metrics Export Function ---
def export_metrics_json(metrics_dict, output_dir="reports", file_name="eval.json"):
    """Safely writes final tracking metrics to a structured JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, file_name)
        with open(full_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Metrics tracking file exported successfully to: {full_path}")
    except Exception as e:
        logger.error(f"Failed to export metrics block to JSON storage: {e}")
        raise

# --- Main Orchestration Execution ---
def main():
    try:
        # 1. Load Evaluation Data
        X_test, y_test_raw = load_test_features("./data/processed/test_tfid.csv")
        
        # 2. Extract Pipeline Assets
        xgb_model, label_encoder = load_pipeline_artifacts(models_dir="models")
        
        # 3. Process Model Scoring
        tracking_metrics = evaluate_model_performance(
            xgb_model, label_encoder, X_test, y_test_raw
        )
        
        # 4. Save Final Performance Log File
        export_metrics_json(
            tracking_metrics, output_dir="reports", file_name="eval.json"
        )
        logger.info("Model evaluation stage pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Evaluation pipeline processing failed completely: {e}")

if __name__ == "__main__":
    main()
