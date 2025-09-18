#!/usr/bin/env python3

"""
This script:
- Queries the BigQuery public dataset bigquery-public-data.ml_datasets.census_adult_income
- Saves two CSVs locally into your repo (data/train.csv and data/test.csv)
"""

import argparse, os
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

# SQL query to fetch relevant columns from the census income dataset
QUERY = '''
SELECT
  age, workclass, education, marital_status, occupation, relationship,
  race, sex, capital_gain, capital_loss, hours_per_week, native_country,
  income_bracket
FROM `bigquery-public-data.ml_datasets.census_adult_income`
'''

def main(project_id: str, out_dir: str = "data"):
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Query BigQuery public dataset
    client = bigquery.Client(project=project_id)
    # Execute the query and convert the result to a pandas DataFrame
    df = client.query(QUERY).result().to_dataframe(create_bqstorage_client=True)

    # Basic clean
    df = df.dropna().copy()
    # Create binary label: 1 if >50K else 0 
    df["label"] = (df["income_bracket"].str.contains(">50K")).astype(int)
    df = df.drop(columns=["income_bracket"])

    # Split into train and test sets (80% train, 20% test) with stratification on the label column to maintain class distribution 
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    # Save train dataset to CSV files
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    # Save test dataset to CSV files
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Wrote {out_dir}/train.csv and {out_dir}/test.csv (rows: {len(train_df)}, {len(test_df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True, help="Your GCP project ID")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()
    main(args.project_id, args.out_dir)
