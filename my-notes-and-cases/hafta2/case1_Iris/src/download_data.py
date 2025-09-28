import pandas as pd
import seaborn as sns
import os

def load_data():
    """Load the Iris dataset and save it as a CSV file."""

    os.makedirs("data/raw", exist_ok=True)

    df = sns.load_dataset("iris")
    df.to_csv("data/raw/iris.csv", index=False)

    print("Data loaded and saved to data/raw/iris.csv")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(df.head())

    return df

if __name__ == "__main__":
    load_data()

