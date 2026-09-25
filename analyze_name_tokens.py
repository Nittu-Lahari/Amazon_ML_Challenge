import pandas as pd
import re
from collections import Counter


def tokenize(name):
    if not isinstance(name, str):
        return []

    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name.split()


s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["business_name"]
)

counter = Counter()

for name in s2["business_name"]:
    tokens = set(tokenize(name))
    counter.update(tokens)

print("\nTotal unique tokens:", len(counter))

print("\nTop 50 most common tokens:\n")

for token, count in counter.most_common(50):
    print(f"{token:30} {count}")