import csv
import os
import re
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "candidate_index.db")
S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
CANDIDATE_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs_v3.tsv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "matching_results_v3.tsv")

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


def tokens(text):
    return {
        x for x in text.split()
        if len(x) >= 3 and x not in STOPWORDS
    }


print("Loading candidate pairs...")

candidates = {}

with open(CANDIDATE_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        value = row["candidate_entity_ids"]

        candidates[row["source1_entity_id"]] = (
            value.split(",") if value else []
        )

print(f"Loaded {len(candidates):,} candidate lists.")


print("Loading records...")

conn = sqlite3.connect(DB_PATH)

record_map = {}

for entity_id, name_norm, address_norm, country in conn.execute(
    "SELECT entity_id, name_norm, address_norm, country FROM records"
):
    record_map[entity_id] = (
        name_norm or "",
        address_norm or "",
        country or ""
    )

conn.close()

print(f"Loaded {len(record_map):,} records.")
print("Starting token matcher...")


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

        s1_name_tokens = tokens(s1_name)
        s1_address_tokens = tokens(s1_address)

        scored = []

        for candidate_id in candidates.get(s1_id, []):

            record = record_map.get(candidate_id)

            if record is None:
                continue

            c_name, c_address, c_country = record

            if c_country != country:
                continue

            c_name_tokens = tokens(c_name)
            c_address_tokens = tokens(c_address)

            # Exact normalized evidence
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
                score = 100

            elif name_exact:
                score = 80

            elif address_exact:
                score = 75

            else:
                # Jaccard token overlap
                if s1_name_tokens and c_name_tokens:
                    name_intersection = (
                        len(s1_name_tokens & c_name_tokens)
                    )
                    name_union = (
                        len(s1_name_tokens | c_name_tokens)
                    )
                    name_jaccard = (
                        name_intersection / name_union
                        if name_union else 0
                    )
                else:
                    name_jaccard = 0

                if s1_address_tokens and c_address_tokens:
                    addr_intersection = (
                        len(s1_address_tokens & c_address_tokens)
                    )
                    addr_union = (
                        len(s1_address_tokens | c_address_tokens)
                    )
                    address_jaccard = (
                        addr_intersection / addr_union
                        if addr_union else 0
                    )
                else:
                    address_jaccard = 0

                score = (
                    60 * name_jaccard +
                    40 * address_jaccard
                )

            scored.append((candidate_id, score))

        final_matches = []

        if scored:

            scored.sort(
                key=lambda x: x[1],
                reverse=True
            )

            best_id, best_score = scored[0]

            # Conservative thresholds
            if best_score >= 70:
                final_matches.append(best_id)

                # Allow only very close high-quality ties
                for candidate_id, score in scored[1:]:

                    if (
                        score >= 85
                        and best_score - score <= 5
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
print("V3 TOKEN matching complete.")
print(f"Total Source 1 records: {total:,}")
print(f"S1 with matches: {matched:,}")
print(f"Output: {OUTPUT_PATH}")