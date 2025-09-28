import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data(input_path = "data/raw/iris.csv", output_path = "data/processed/iris_processed.csv"):

    # Load the dataset
    df = pd.read_csv(input_path)

    df['species'] = LabelEncoder().fit_transform(df['species'])

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    

    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")
    print(f"Processed data shape: {df.shape}")
    print(df.head())
    print("Data preprocessing completed.")

    print(f"Features: {feature_cols}")

    return df,feature_cols

if __name__ == "__main__":
    preprocess_data()
