import pandas as pd
import os

train_files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv"
]

test_files = [
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]


def inspect_file(file_path):
    print("\n" + "=" * 60)
    print("FILE:", file_path)
    print("=" * 60)

    df = pd.read_csv(file_path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    return df


print("\n\n========== TRAINING DATA ==========")

for file in train_files:
    inspect_file(file)


print("\n\n========== TESTING DATA ==========")

for file in test_files:
    inspect_file(file)