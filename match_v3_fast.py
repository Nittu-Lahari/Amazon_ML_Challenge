import csv
import os
import re
import sqlite3
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "candidate_index.db")
S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
CANDIDATE_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs_v3.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "matching_results_v3.tsv")


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


print("Loading candidate pairs...")

candidates = {}

with open(CANDIDATE_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        ids = row["candidate_entity_ids"]

        candidates[row["source1_entity_id"]] = (
            ids.split(",") if ids else []
        )

print(f"Loaded {len(candidates):,} candidate lists.")


print("Loading records into memory...")

conn = sqlite3.connect(DB_PATH)

record_map = {}

cursor = conn.execute(
    "SELECT entity_id, name_norm, address_norm, country FROM records"
)

count = 0

for entity_id, name_norm, address_norm, country in cursor:

    record_map[entity_id] = (
        name_norm or "",
        address_norm or "",
        country or ""
    )

    count += 1

    if count % 1_000_000 == 0:
        print(f"Loaded records: {count:,}")

conn.close()

print(f"Loaded records: {count:,}")
print("Starting V3 matching...")


total = 0
matched = 0


with open(S1_PATH, "r", encoding="utf-8", newline="") as s1_file, \
     open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out_file:

    reader = csv.DictReader(s1_file, delimiter="\t")

    writer = csv.writer(out_file, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    for row in reader:

        total += 1

        s1_id = row["entity_id"]
        country = row["country"] or ""

        s1_name = normalize(row["business_name"])
        s1_address = normalize(row["business_address"])

        candidate_ids = candidates.get(s1_id, [])

        # --------------------------------------------------
        # Stage 1: exact evidence
        # --------------------------------------------------

        exact_both = []
        exact_name = []
        exact_address = []

        for candidate_id in candidate_ids:

            record = record_map.get(candidate_id)

            if record is None:
                continue

            c_name, c_address, c_country = record

            if c_country != country:
                continue

            name_exact = (
                bool(s1_name)
                and bool(c_name)
                and s1_name == c_name
            )

            address_exact = (
                bool(s1_address)
                and bool(c_address)
                and s1_address == c_address
            )

            if name_exact and address_exact:
                exact_both.append(candidate_id)

            elif name_exact:
                exact_name.append(candidate_id)

            elif address_exact:
                exact_address.append(candidate_id)

        final_matches = []

        # Strongest evidence
        if exact_both:
            final_matches = exact_both

        # A unique exact name is strong evidence
        elif len(exact_name) == 1:
            final_matches = exact_name

        # A unique exact address is strong evidence
        elif len(exact_address) == 1:
            final_matches = exact_address

        # --------------------------------------------------
        # Stage 2: similarity only when exact evidence
        # is inconclusive
        # --------------------------------------------------

        elif len(candidate_ids) <= 40:

            scored = []

            for candidate_id in candidate_ids:

                record = record_map.get(candidate_id)

                if record is None:
                    continue

                c_name, c_address, c_country = record

                if c_country != country:
                    continue

                name_score = similarity(s1_name, c_name)
                address_score = similarity(s1_address, c_address)

                if s1_name and s1_address:
                    score = (
                        0.60 * name_score +
                        0.40 * address_score
                    )
                elif s1_name:
                    score = name_score
                elif s1_address:
                    score = address_score
                else:
                    score = 0.0

                scored.append((candidate_id, score))

            if scored:

                scored.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                best_id, best_score = scored[0]

                # Conservative threshold
                if best_score >= 0.88:
                    final_matches = [best_id]

                    # Keep close ties only when extremely strong
                    for candidate_id, score in scored[1:]:

                        if (
                            score >= 0.94
                            and best_score - score <= 0.03
                        ):
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


print()
print("V3 FAST matching complete.")
print(f"Total Source 1 records: {total:,}")
print(f"S1 with matches: {matched:,}")
print(f"Output: {OUTPUT_PATH}")