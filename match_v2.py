import csv
import os
import re
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "candidate_index.db")
S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
CANDIDATE_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "matching_results_v2.tsv")


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


# Load candidate pairs
print("Loading candidate pairs...")

candidates = {}

with open(CANDIDATE_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        ids = row["candidate_entity_ids"]

        if ids:
            candidates[row["source1_entity_id"]] = ids.split(",")
        else:
            candidates[row["source1_entity_id"]] = []


print(f"Loaded candidates for {len(candidates):,} Source-1 records.")


# Load candidate record details from SQLite
import sqlite3

conn = sqlite3.connect(DB_PATH)

print("Loading Source-1 records...")

with open(S1_PATH, "r", encoding="utf-8", newline="") as s1_file, \
     open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_file:

    reader = csv.DictReader(s1_file, delimiter="\t")
    writer = csv.writer(out_file, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    total = 0
    matched = 0

    for row in reader:
        total += 1

        s1_id = row["entity_id"]
        country = row["country"]

        s1_name = normalize(row["business_name"])
        s1_address = normalize(row["business_address"])

        candidate_ids = candidates.get(s1_id, [])

        scored = []

        for candidate_id in candidate_ids:

            result = conn.execute(
                """
                SELECT name_norm, address_norm
                FROM records
                WHERE entity_id = ? AND country = ?
                """,
                (candidate_id, country)
            ).fetchone()

            if result is None:
                continue

            c_name, c_address = result

            name_score = similarity(s1_name, c_name)
            address_score = similarity(s1_address, c_address)

            # Strong exact evidence
            if s1_name and s1_address:
                if s1_name == c_name and s1_address == c_address:
                    score = 1.0
                else:
                    score = 0.60 * name_score + 0.40 * address_score

            elif s1_name:
                score = name_score

            elif s1_address:
                score = address_score

            else:
                score = 0.0

            scored.append((candidate_id, score))

        # Conservative V2 decision
        final_matches = []

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)

            best_id, best_score = scored[0]

            # Accept only strong matches.
            if best_score >= 0.88:
                final_matches.append(best_id)

                # Allow additional very-close matches.
                for candidate_id, score in scored[1:]:
                    if score >= 0.94 and best_score - score <= 0.03:
                        final_matches.append(candidate_id)

        if final_matches:
            matched += 1

        writer.writerow([
            s1_id,
            ",".join(sorted(set(final_matches)))
        ])

        if total % 100000 == 0:
            print(
                f"Processed: {total:,} | "
                f"S1 with matches: {matched:,}"
            )

conn.close()

print()
print("V2 matching complete.")
print(f"Total Source 1 records: {total:,}")
print(f"Source 1 records with matches: {matched:,}")
print(f"Output: {OUTPUT_PATH}")