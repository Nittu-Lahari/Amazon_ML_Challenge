import pandas as pd
import re


def normalize_text(value):
    if not isinstance(value, str):
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


# ---------- SOURCE 1 ----------
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s1["name_norm"] = s1["business_name"].map(normalize_text)
s1["address_norm"] = s1["business_address"].map(normalize_text)

s1_keys = {}

for row in s1.itertuples(index=False):
    country = str(row.country)

    s1_keys[row.entity_id] = (
        country + "|" + row.name_norm,
        country + "|" + row.address_norm
    )

del s1


# ---------- SOURCE 2 ----------
s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name", "business_address", "country"]
)

s2["name_norm"] = s2["business_name"].map(normalize_text)
s2["address_norm"] = s2["business_address"].map(normalize_text)

s2_keys = {}

for row in s2.itertuples(index=False):
    country = str(row.country)

    s2_keys[row.entity_id] = (
        country + "|" + row.name_norm,
        country + "|" + row.address_norm
    )

del s2


# ---------- GROUND TRUTH ----------
gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

total = 0
found_name = 0
found_address = 0
found_combined = 0

for row in gt.itertuples(index=False):

    matched = row.matched_entity_ids

    if pd.isna(matched) or str(matched).strip() == "":
        continue

    s1_info = s1_keys.get(row.source1_entity_id)

    if s1_info is None:
        continue

    s1_name_key, s1_address_key = s1_info

    for s2_id in str(matched).split(","):

        s2_id = s2_id.strip()

        s2_info = s2_keys.get(s2_id)

        if s2_info is None:
            continue

        s2_name_key, s2_address_key = s2_info

        total += 1

        name_match = (
            s1_name_key != "nan|" and
            s1_name_key == s2_name_key
        )

        address_match = (
            s1_address_key != "nan|" and
            s1_address_key == s2_address_key
        )

        if name_match:
            found_name += 1

        if address_match:
            found_address += 1

        if name_match or address_match:
            found_combined += 1


print("\nTotal ground-truth S2 matches:", total)

print("\nName block:", found_name)
print("Name recall:", found_name / total)

print("\nAddress block:", found_address)
print("Address recall:", found_address / total)

print("\nCombined block:", found_combined)
print("Combined recall:", found_combined / total)