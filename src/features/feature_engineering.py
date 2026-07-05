import numpy as np 
import pandas as pd
import os 
from sklearn.feature_extraction.text import CountVectorizer
import yaml 
max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
# Fetch the data from data/processed
train_data = pd.read_csv('./data/interim/train_interim.csv')
test_data = pd.read_csv('./data/interim/test_interim.csv')

# FIX 1: Fill the np.nan values created in preprocessing with empty strings ""
train_data['content'] = train_data['content'].fillna("")
test_data['content'] = test_data['content'].fillna("")

# Extract values
X_train = train_data["content"].values
y_train = train_data["sentiment"].values

X_test = test_data["content"].values
y_test = test_data["sentiment"].values

# Initialize vectorizer
vectorizer = CountVectorizer(max_features=5000)

# Fit and transform
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Build feature DataFrames
train_df = pd.DataFrame(X_train_bow.toarray())
train_df["label"] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df["label"] = y_test  # FIX 2: Changed y_train to y_test here

# Create output folder
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

# FIX 3: Save train_df and test_df (the features), NOT train_data
train_df.to_csv(os.path.join(data_path, "train_count.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_count.csv"), index=False)

print("Feature engineering completed successfully!")
