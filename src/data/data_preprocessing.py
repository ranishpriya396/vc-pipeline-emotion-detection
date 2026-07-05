import logging
import os
import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- Environment Setup Function ---
def initialize_nlp_resources():
    """Safely downloads required NLTK resources."""
    try:
        logger.info("Downloading NLTK resources...")
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        logger.info("NLTK resources ready.")
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {e}")
        raise


# --- Data Loading Function ---
def load_dataset(file_path):
    """Loads a CSV file into a pandas DataFrame with error handling."""
    try:
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise


# --- Text Cleaning Helper Functions ---
def lower_case(text):
    return " ".join([word.lower() for word in str(text).split()])


def remove_stop_words(text, stop_words):
    return " ".join(
        [word for word in str(text).split() if word not in stop_words]
    )


def removing_numbers(text):
    return "".join([char for char in str(text) if not char.isdigit()])


def removing_punctuations(text):
    text = re.sub(r"[^\w\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def removing_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))


def lemmatization(text, lemmatizer):
    return " ".join(
        [lemmatizer.lemmatize(word) for word in str(text).split()]
    )


def remove_small_sentences(df):
    """Replaces sentences with fewer than 3 words with NaN."""
    try:
        word_counts = df["content"].astype(str).str.split().str.len()
        df.loc[word_counts < 3, "content"] = np.nan
        return df
    except KeyError:
        logger.error(
            "Column 'content' not found during short sentence removal."
        )
        raise
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}")
        raise


# --- Core Pipeline Function ---
def normalize_text(df, lemmatizer, stop_words):
    """Executes the sequential text normalization pipeline."""
    if "content" not in df.columns:
        logger.error("DataFrame missing required 'content' column.")
        raise KeyError("Column 'content' is missing.")

    try:
        df = df.copy()
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(removing_urls)
        df["content"] = df["content"].apply(removing_numbers)
        df["content"] = df["content"].apply(removing_punctuations)
        df["content"] = df["content"].apply(
            lambda x: remove_stop_words(x, stop_words)
        )
        df["content"] = df["content"].apply(
            lambda x: lemmatization(x, lemmatizer)
        )
        df = remove_small_sentences(df)
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


# --- Data Saving Function ---
def save_processed_data(df, output_dir, file_name):
    """Safely creates output directories and saves the DataFrame."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, file_name)
        df.to_csv(full_path, index=False)
        logger.info(f"Successfully saved processed data to {full_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_name}: {e}")
        raise


# --- Main Orchestration Execution ---
def main():
    try:
        # Initialize resources
        initialize_nlp_resources()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        # Load datasets
        train_data = load_dataset("./data/raw/train.csv")
        test_data = load_dataset("./data/raw/test.csv")

        # Process datasets
        logger.info("Processing training data...")
        train_processed_data = normalize_text(train_data, lemmatizer, stop_words)

        logger.info("Processing testing data...")
        test_processed_data = normalize_text(test_data, lemmatizer, stop_words)

        # Save outputs
        output_directory = os.path.join("data", "interim")
        save_processed_data(
            train_processed_data, output_directory, "train_interim.csv"
        )
        save_processed_data(
            test_processed_data, output_directory, "test_interim.csv"
        )

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
