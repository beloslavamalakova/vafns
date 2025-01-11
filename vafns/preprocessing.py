"""
Date: 11 Jan 2025
Author: Beloslava Malakova

Notes:
- Used dataset has data till 2023 incl.

"""
import pandas as pd
import numpy as np

def preprocess_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Initial Dataset Info:")
    print(df.info())

    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Drop rows with more than 30% missing values
    df = df.dropna(axis=0, thresh=int(0.7 * len(df.columns)))

    # Drop columns with more than 30% missing values
    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))

    # Fill remaining NaN values with the mean of numeric columns only
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Interpolate for numerical columns (if applicable)
    df = df.interpolate(method='linear', inplace=False)

    # Verify if any NaN values remain
    if df.isnull().sum().sum() == 0:
        print("All missing values handled successfully!")
    else:
        print("Some missing values remain. Inspect the dataset.")

    # Convert Date column to datetime format
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
            print("Date column converted to datetime format.")
        except Exception as e:
            print(f"Error converting Date column: {e}")

    try:
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
    except Exception as e:
        print(f"Error saving cleaned dataset: {e}")

# Specify the input and output file paths
input_file = 'yield-curve-rates.csv'  # Replace with your dataset file
output_file = 'cleaned_dataset.csv'  # Replace with the desired output file

preprocess_dataset(input_file, output_file)
