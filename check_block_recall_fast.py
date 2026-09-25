import pandas as pd
import re

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------- SOURCE 1 ----------
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s1["name_norm"] = s1["business_name"].map(normalize_name)

# Direct dictionary lookup: S1 ID -> block key
s1_keys = dict(
    zip(
        s1["entity_id"],
        s1["country"].astype(str) + "|" + s1["name_norm"]
    )
)

del s1


# ---------- SOURCE 2 ----------
s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "country"]
)

s2["name_norm"] = s2["business_name"].map(normalize_name)

# Direct dictionary lookup: S2 ID -> block key
s2_keys = dict(
    zip(
        s2["entity_id"],
        s2["country"].astype(str) + "|" + s2["name_norm"]
    )
)

del s2


# ---------- GROUND TRUTH ----------
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

    s1_key = s1_keys.get(row.source1_entity_id)

    if s1_key is None:
        continue

    for s2_id in str(matched).split(","):

        s2_id = s2_id.strip()

        if s2_id in s2_keys:

            total += 1

            if s2_keys[s2_id] == s1_key:
                found += 1


print("\nTotal ground-truth S2 matches:", total)
print("Found by exact name blocking:", found)

if total:
    print("Blocking recall:", found / total)