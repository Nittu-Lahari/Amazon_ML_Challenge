import csv
import os
import re
import sqlite3
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "candidate_index.db")
S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
OLD_CANDIDATES = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs_v3.tsv")

MAX_PER_TOKEN = 20
MAX_NEW_PER_S1 = 60


STOPWORDS = {
    "the", "and", "for", "with", "from",
    "llc", "inc", "ltd", "limited",
    "private", "pvt", "corp", "corporation",
    "company", "co", "plc", "group",
    "services", "india", "usa", "united"
}


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def informative_tokens(name):
    return {
        token
        for token in name.split()
        if len(token) >= 4 and token not in STOPWORDS
    }


print("Reading Source 1 names...")

s1_tokens = {}

with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        name = normalize(row["business_name"])
        s1_tokens[row["entity_id"]] = informative_tokens(name)

print(f"Loaded {len(s1_tokens):,} Source-1 records.")


print("Building token -> Source-1 map...")

token_to_s1 = defaultdict(list)

for s1_id, tokens in s1_tokens.items():
    for token in tokens:
        token_to_s1[token].append(s1_id)

print(f"Unique relevant tokens: {len(token_to_s1):,}")


print("Scanning database for token candidates...")

conn = sqlite3.connect(DB_PATH)

new_candidates = defaultdict(set)
token_counts = defaultdict(int)

cursor = conn.execute(
    """
    SELECT entity_id, name_norm, country
    FROM records
    WHERE name_norm != ''
    """
)

processed = 0

for entity_id, name_norm, country in cursor:

    processed += 1

    tokens = informative_tokens(name_norm)

    for token in tokens:

        s1_list = token_to_s1.get(token)

        if not s1_list:
            continue

        # Keep only a bounded number of records per token.
        if token_counts[(country, token)] >= MAX_PER_TOKEN:
            continue

        token_counts[(country, token)] += 1

        for s1_id in s1_list:

            if len(new_candidates[s1_id]) >= MAX_NEW_PER_S1:
                continue

            new_candidates[s1_id].add(entity_id)

    if processed % 1_000_000 == 0:
        print(
            f"Database records scanned: {processed:,} | "
            f"S1 with new candidates: {len(new_candidates):,}"
        )

conn.close()

print()
print(f"Database records scanned: {processed:,}")
print(f"S1 with new candidates: {len(new_candidates):,}")


print("Merging with existing V2 candidate pairs...")

with open(OLD_CANDIDATES, "r", encoding="utf-8", newline="") as old_file, \
     open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_file:

    reader = csv.DictReader(old_file, delimiter="\t")
    writer = csv.writer(out_file, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    total = 0
    expanded = 0

    for row in reader:

        total += 1

        s1_id = row["source1_entity_id"]

        old_ids = (
            set(row["candidate_entity_ids"].split(","))
            if row["candidate_entity_ids"]
            else set()
        )

        added_ids = new_candidates.get(s1_id, set())

        combined = old_ids | added_ids

        if added_ids:
            expanded += 1

        writer.writerow([
            s1_id,
            ",".join(sorted(combined))
        ])

        if total % 100000 == 0:
            print(
                f"Merged: {total:,} | "
                f"S1 expanded: {expanded:,}"
            )


print()
print("V3 candidate generation complete.")
print(f"Total S1 rows: {total:,}")
print(f"S1 rows expanded: {expanded:,}")
print(f"Output: {OUTPUT_PATH}")