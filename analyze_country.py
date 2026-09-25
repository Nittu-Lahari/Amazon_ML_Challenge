import pandas as pd
import os

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv"
]

for file in files:
    print("\n" + "=" * 50)
    print(file)

    if file.startswith("train_"):
        path = os.path.join("dataset", "train", file)
    else:
        path = os.path.join("dataset", "test", file)

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["country"]
    )

    print(df["country"].value_counts(dropna=False))