import logging
import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- Configuration Loader ---
def load_model_params(params_path="params.yaml"):
    """Loads model hyperparameters from a YAML file with fallbacks."""
    try:
        logger.info(f"Loading parameters from {params_path}...")
        with open(params_path, "r") as file:
            config = yaml.safe_load(file)
        params = config["model_building"]
        return params
    except FileNotFoundError:
        logger.warning(f"{params_path} not found. Using default parameters.")
        return {"n_estimators": 100, "learning_rate": 0.1}
    except KeyError:
        logger.warning(
            "Key 'model_building' missing in YAML. Using default parameters."
        )
        return {"n_estimators": 100, "learning_rate": 0.1}
    except Exception as e:
        logger.error(f"Error reading parameters file: {e}")
        raise


# --- Data Preparation Function ---
def load_and_split_features(file_path):
    """Loads feature file and separates independent features from the target."""
    try:
        logger.info(f"Loading training features from {file_path}...")
        train_data = pd.read_csv(file_path)

        if train_data.empty:
            raise ValueError(f"The dataset at {file_path} is empty.")

        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        logger.info(f"Loaded feature matrix shape: {X_train.shape}")
        return X_train, y_train
    except FileNotFoundError:
        logger.error(f"Feature file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error during feature loading: {e}")
        raise


# --- Label Encoding Function ---
def encode_target_labels(y_train):
    """Encodes target labels to integers for XGBoost compatibility."""
    try:
        logger.info("Encoding target labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)

        num_classes = len(np.unique(y_train_encoded))
        logger.info(f"Detected {num_classes} unique target classes.")

        return y_train_encoded, label_encoder, num_classes
    except Exception as e:
        logger.error(f"Failed to encode target labels: {e}")
        raise


# --- Model Training Function ---
def train_xgboost(X_train, y_train_encoded, num_classes, params):
    """Configures objective settings dynamically and trains the XGBoost model."""
    try:
        # Determine classification settings based on unique classes
        if num_classes <= 2:
            objective_setting = "binary:logistic"
            num_class_setting = None
            logger.info("Configuring XGBoost for Binary Classification.")
        else:
            objective_setting = "multi:softmax"
            num_class_setting = num_classes
            logger.info("Configuring XGBoost for Multiclass Classification.")

        # Initialize model with user and explicit default parameters
        xgb_model = XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=8,
            learning_rate=params.get("learning_rate", 0.1),
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective_setting,
            num_class=num_class_setting,
            random_state=42,
        )

        logger.info("Training XGBoost classifier...")
        xgb_model.fit(X_train, y_train_encoded)
        logger.info("Model training completed successfully.")

        return xgb_model
    except Exception as e:
        logger.error(f"Error during model training execution: {e}")
        raise


# --- Artifact Export Function ---
def save_model_artifacts(model, encoder, output_dir="models"):
    """Saves the trained model and label encoder objects to disk."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "model.pkl")
        encoder_path = os.path.join(output_dir, "label_encoder.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
        logger.info(f"Label encoder saved to {encoder_path}")

    except Exception as e:
        logger.error(f"Failed to save training artifacts: {e}")
        raise


# --- Main Orchestration Execution ---
def main():
    try:
        # 1. Load Configurations
        params = load_model_params("params.yaml")

        # 2. Extract Data
        X_train, y_train = load_and_split_features("./data/processed/train_tfid.csv")

        # 3. Transform Targets
        y_train_encoded, label_encoder, num_classes = encode_target_labels(
            y_train
        )

        # 4. Fit Model
        xgb_model = train_xgboost(X_train, y_train_encoded, num_classes, params)

        # 5. Export Assets
        save_model_artifacts(xgb_model, label_encoder, output_dir="models")

        logger.info("Model training stage pipeline executed successfully.")

    except Exception as e:
        logger.critical(f"Model training pipeline failed completely: {e}")


if __name__ == "__main__":
    main()
