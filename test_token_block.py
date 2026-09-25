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


# ---------------- SOURCE 1 ----------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s1["tokens"] = s1["business_name"].map(tokenize)


# ---------------- SOURCE 2 ----------------

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s2["tokens"] = s2["business_name"].map(tokenize)


# ---------------- FIND TOKEN FREQUENCY ----------------

counter = Counter()

for tokens in s2["tokens"]:
    counter.update(set(tokens))


# Ignore very common tokens
common_tokens = {
    token for token, count in counter.items()
    if count > 100000
}

print("Tokens removed:", len(common_tokens))


# ---------------- CREATE S1 BLOCK KEYS ----------------

s1_blocks = {}

for row in s1.itertuples(index=False):

    useful_tokens = [
        token
        for token in set(row.tokens)
        if token not in common_tokens
    ]

    s1_blocks[row.entity_id] = {
        str(row.country) + "|" + token
        for token in useful_tokens
    }


# ---------------- CREATE S2 INDEX ----------------

s2_index = {}

for row in s2.itertuples(index=False):

    useful_tokens = [
        token
        for token in set(row.tokens)
        if token not in common_tokens
    ]

    for token in useful_tokens:

        key = str(row.country) + "|" + token

        if key not in s2_index:
            s2_index[key] = set()

        s2_index[key].add(row.entity_id)


del s1
del s2


# ---------------- GROUND TRUTH ----------------

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


total = 0
found = 0

for row in gt.itertuples(index=False):

    matched = row.matched_entity_ids

    if pd.isna(matched) or str(matched).strip() == "":
        continue

    blocks = s1_blocks.get(row.source1_entity_id, set())

    if not blocks:
        continue

    for s2_id in str(matched).split(","):

        s2_id = s2_id.strip()

        if not s2_id:
            continue

        total += 1

        # Check whether this true S2 record
        # exists in ANY S1 token block
        found_here = False

        for block in blocks:

            if s2_id in s2_index.get(block, set()):
                found_here = True
                break

        if found_here:
            found += 1


print("\nTotal ground-truth S2 matches:", total)
print("Found by token blocking:", found)

if total:
    print("Token blocking recall:", found / total)