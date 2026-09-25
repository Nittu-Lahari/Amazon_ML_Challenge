import pandas as pd
import re
from collections import defaultdict


def normalize_text(value):
    if not isinstance(value, str):
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def load_source(path):
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    df["name_norm"] = df["business_name"].map(normalize_text)
    df["address_norm"] = df["business_address"].map(normalize_text)

    return df


def evaluate(source1_path, source_path, source_label):

    print("\n" + "=" * 60)
    print(source_label)
    print("=" * 60)

    # Source 1
    s1 = load_source(source1_path)

    s1_name = dict(
        zip(
            s1["entity_id"],
            s1["country"].astype(str) + "|" + s1["name_norm"]
        )
    )

    s1_address = dict(
        zip(
            s1["entity_id"],
            s1["country"].astype(str) + "|" + s1["address_norm"]
        )
    )

    del s1

    # Source 2 / Source 3
    sx = load_source(source_path)

    sx_name = dict(
        zip(
            sx["entity_id"],
            sx["country"].astype(str) + "|" + sx["name_norm"]
        )
    )

    sx_address = dict(
        zip(
            sx["entity_id"],
            sx["country"].astype(str) + "|" + sx["address_norm"]
        )
    )

    del sx

    # Ground truth
    gt = pd.read_csv(
        "dataset/train/train_ground_truth.tsv",
        sep="\t"
    )

    total = 0
    name_found = 0
    address_found = 0
    combined_found = 0

    for row in gt.itertuples(index=False):

        matched = row.matched_entity_ids

        if pd.isna(matched) or str(matched).strip() == "":
            continue

        s1_id = row.source1_entity_id

        name_key = s1_name.get(s1_id)
        address_key = s1_address.get(s1_id)

        if name_key is None:
            continue

        for target_id in str(matched).split(","):

            target_id = target_id.strip()

            if not target_id:
                continue

            target_name = sx_name.get(target_id)
            target_address = sx_address.get(target_id)

            if target_name is None:
                continue

            total += 1

            name_match = (
                name_key == target_name
                and name_key != "nan|"
            )

            address_match = (
                address_key == target_address
                and address_key != "nan|"
            )

            if name_match:
                name_found += 1

            if address_match:
                address_found += 1

            if name_match or address_match:
                combined_found += 1

    print("Total ground-truth matches:", total)

    print(
        "Name block recall:",
        name_found / total if total else 0
    )

    print(
        "Address block recall:",
        address_found / total if total else 0
    )

    print(
        "Combined recall:",
        combined_found / total if total else 0
    )


# S2
evaluate(
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "SOURCE 1 → SOURCE 2"
)

# S3
evaluate(
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source3.tsv",
    "SOURCE 1 → SOURCE 3"
)