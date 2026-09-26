import csv
import os
import re
import sqlite3
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "train_candidate_index.db")

S1_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source1.tsv"
)

GT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    BASE_DIR,
    "output",
    "train_candidate_pairs.tsv"
)


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def similarity(a, b):
    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(a, b):
    ta = set(a.split())
    tb = set(b.split())

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


print("Loading Source-1 training records...")

s1 = {}

with open(
    S1_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    for row in csv.DictReader(f, delimiter="\t"):

        s1[row["entity_id"]] = (
            row["country"] or "",
            normalize(row["business_name"]),
            normalize(row["business_address"])
        )

print(f"Loaded S1 records: {len(s1):,}")


print("Loading ground truth...")

ground_truth = {}

with open(
    GT_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    for row in csv.DictReader(f, delimiter="\t"):

        ids = row["matched_entity_ids"]

        ground_truth[row["source1_entity_id"]] = (
            set(ids.split(",")) if ids else set()
        )

print(
    f"Loaded ground-truth rows: "
    f"{len(ground_truth):,}"
)


print("Loading candidate pairs...")

candidates = {}

with open(
    CANDIDATE_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    for row in csv.DictReader(f, delimiter="\t"):

        ids = row["candidate_entity_ids"]

        candidates[row["source1_entity_id"]] = (
            ids.split(",") if ids else []
        )

print(
    f"Loaded candidate lists: "
    f"{len(candidates):,}"
)


print("Loading candidate records from SQLite...")

needed_ids = set()

for ids in candidates.values():
    needed_ids.update(ids)

print(
    f"Unique candidate IDs: "
    f"{len(needed_ids):,}"
)

conn = sqlite3.connect(DB_PATH)

record_map = {}

cursor = conn.execute(
    """
    SELECT
        entity_id,
        name_norm,
        address_norm,
        country
    FROM records
    """
)

for entity_id, name_norm, address_norm, country in cursor:

    if entity_id in needed_ids:

        record_map[entity_id] = (
            name_norm or "",
            address_norm or "",
            country or ""
        )

conn.close()

print(
    f"Loaded candidate records: "
    f"{len(record_map):,}"
)


print()
print("Analyzing TRUE matches vs FALSE candidates...")
print()

stats = {
    "true": {
        "count": 0,
        "name_seq": 0.0,
        "addr_seq": 0.0,
        "name_jac": 0.0,
        "addr_jac": 0.0,
    },
    "false": {
        "count": 0,
        "name_seq": 0.0,
        "addr_seq": 0.0,
        "name_jac": 0.0,
        "addr_jac": 0.0,
    }
}

processed = 0

for s1_id, true_ids in ground_truth.items():

    if not true_ids:
        continue

    s1_row = s1.get(s1_id)

    if s1_row is None:
        continue

    country, s1_name, s1_address = s1_row

    candidate_ids = candidates.get(s1_id, [])

    true_set = true_ids

    for candidate_id in candidate_ids:

        record = record_map.get(candidate_id)

        if record is None:
            continue

        c_name, c_address, c_country = record

        if c_country != country:
            continue

        name_seq = similarity(
            s1_name,
            c_name
        )

        addr_seq = similarity(
            s1_address,
            c_address
        )

        name_jac = token_jaccard(
            s1_name,
            c_name
        )

        addr_jac = token_jaccard(
            s1_address,
            c_address
        )

        kind = (
            "true"
            if candidate_id in true_set
            else "false"
        )

        stats[kind]["count"] += 1
        stats[kind]["name_seq"] += name_seq
        stats[kind]["addr_seq"] += addr_seq
        stats[kind]["name_jac"] += name_jac
        stats[kind]["addr_jac"] += addr_jac

    processed += 1

    if processed % 100000 == 0:
        print(
            f"Processed S1: {processed:,}"
        )


print()
print("RESULTS")
print("=======")

for kind in ["true", "false"]:

    count = stats[kind]["count"]

    print()
    print(
        kind.upper(),
        "candidate pairs:",
        f"{count:,}"
    )

    if count == 0:
        continue

    print(
        "Average name SequenceMatcher:",
        f"{stats[kind]['name_seq'] / count:.4f}"
    )

    print(
        "Average address SequenceMatcher:",
        f"{stats[kind]['addr_seq'] / count:.4f}"
    )

    print(
        "Average name token Jaccard:",
        f"{stats[kind]['name_jac'] / count:.4f}"
    )

    print(
        "Average address token Jaccard:",
        f"{stats[kind]['addr_jac'] / count:.4f}"
    )


print()
print("Analysis complete.")