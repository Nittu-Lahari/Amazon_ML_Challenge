import pandas as pd
import re

def normalize_name(name):
    if not isinstance(name, str):
        return ""

    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# Read only required columns
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

# Normalize names
s1["name_norm"] = s1["business_name"].map(normalize_name)
s2["name_norm"] = s2["business_name"].map(normalize_name)

# Create blocking key
s1["block_key"] = s1["country"].astype(str) + "|" + s1["name_norm"]
s2["block_key"] = s2["country"].astype(str) + "|" + s2["name_norm"]

print("S1 records:", len(s1))
print("S2 records:", len(s2))

print("\nS1 unique blocking keys:", s1["block_key"].nunique())
print("S2 unique blocking keys:", s2["block_key"].nunique())

print("\nExample duplicate blocking keys in S2:")
print(
    s2["block_key"]
    .value_counts()
    .loc[lambda x: x > 1]
    .head(10)
)