import numpy as np 
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging 
import sys

# LOGGING CONFIGURATION 
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if script is run repeatedly
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('error.log')
    file_handler.setLevel('ERROR')
    file_handler.setFormatter(formatter)
    

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        logger.debug(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)
        test_size = config['data_ingestion']['test_size']
        logger.info(f"Parameters successfully loaded. Test size: {test_size}")
        return float(test_size)
    except FileNotFoundError:
        logger.error(f"Parameter file not found at path: {params_path}")
        raise
    except KeyError:
        logger.error(f"Keys 'data_ingestion' or 'test_size' missing inside {params_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {str(e)}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        logger.debug(f"Reading raw CSV data from: {url}")
        df = pd.read_csv(url)
        logger.info(f"Data successfully loaded. Shape: {df.shape}")
        return df 
    except FileNotFoundError:
        logger.error(f"The raw CSV file was not found at path: {url}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("The target data file is completely empty.")
        raise
    except Exception as e:
        logger.error(f"Failed to read raw data: {str(e)}")
        raise

def processed_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.debug("Starting data processing stage...")
        
        # Safe copy to avoid setting-with-copy warnings
        final_df = df.drop(columns=['tweet_id', 'author'], errors='ignore').copy()
        
        # Filter for targeted sentiment classes
        final_df = final_df[final_df["sentiment"].isin(["neutral", "sadness"])].copy()
        
        # Explicit target mapping instead of inplace modify warnings
        final_df["sentiment"] = final_df["sentiment"].map({"neutral": 1, "sadness": 0})
        
        logger.info(f"Data filtering complete. Filtered dataset shape: {final_df.shape}")
        return final_df  
    except KeyError as e:
        logger.error(f"Required data columns missing during processing: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        logger.debug(f"Attempting to save train/test distributions into: {data_path}")
        
        # FIX: exist_ok=True prevents pipeline crash if folder already exists
        os.makedirs(data_path, exist_ok=True)
        
        # FIX: index=False keeps datasets completely clean for DVC tracking
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        
        logger.info("Train and test files successfully written to disk.")
    except Exception as e:
        logger.error(f"Failed to save datasets to storage: {str(e)}")
        raise

def main():
    try:
        logger.info("=== Starting Data Ingestion Pipeline ===")
        
        test_size = load_params('params.yaml')
        df = read_data(r"C:\Users\HOME\OneDrive\Desktop\Sentiment_Analysis.csv")
        final_df = processed_data(df)
        
        logger.debug("Executing train-test split operations...")
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        
        logger.info("=== Data Ingestion Completed Successfully ===")
    except Exception as e:
        logger.critical(f"Data Ingestion Pipeline FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
