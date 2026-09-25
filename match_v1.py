import sqlite3
import csv
import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "candidate_index.db")
S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_candidate_count(conn, country, field, value):
    if not value:
        return 0

    query = f"""
        SELECT COUNT(*)
        FROM records
        WHERE country = ?
        AND {field} = ?
        AND {field} != ''
    """

    return conn.execute(query, (country, value)).fetchone()[0]


def get_candidates(conn, country, field, value):
    if not value:
        return []

    query = f"""
        SELECT entity_id
        FROM records
        WHERE country = ?
        AND {field} = ?
        AND {field} != ''
    """

    return [row[0] for row in conn.execute(query, (country, value))]


os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

print("Database connected.")
print("Reading Source 1...")

with open(S1_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        total = 0
        matched = 0

        for row in reader:
            total += 1

            s1_id = row["entity_id"]
            country = row["country"]

            name = normalize(row["business_name"])
            address = normalize(row["business_address"])

            name_candidates = []
            address_candidates = []

            # Exact normalized name
            if name:
                count = get_candidate_count(
                    conn,
                    country,
                    "name_norm",
                    name
                )

                # Conservative: use name matches only when the
                # normalized name is reasonably selective.
                if count <= 20:
                    name_candidates = get_candidates(
                        conn,
                        country,
                        "name_norm",
                        name
                    )

            # Exact normalized address
            if address:
                count = get_candidate_count(
                    conn,
                    country,
                    "address_norm",
                    address
                )

                # Conservative: use address matches only when selective.
                if count <= 10:
                    address_candidates = get_candidates(
                        conn,
                        country,
                        "address_norm",
                        address
                    )

            # Combine evidence.
            name_set = set(name_candidates)
            address_set = set(address_candidates)

            # Strongest evidence:
            # candidate matching BOTH exact normalized name and address
            both = name_set & address_set

            if both:
                final_matches = sorted(both)
            else:
                # If only one strong exact field identifies a small
                # candidate set, retain it.
                combined = name_set | address_set

                if len(combined) == 1:
                    final_matches = sorted(combined)
                else:
                    final_matches = []

            if final_matches:
                matched += 1

            writer.writerow([
                s1_id,
                ",".join(final_matches)
            ])

            if total % 100000 == 0:
                print(
                    f"Processed: {total:,} | "
                    f"S1 with matches: {matched:,}"
                )

conn.close()

print()
print("Matching complete.")
print(f"Total Source 1 records: {total:,}")
print(f"Source 1 records with matches: {matched:,}")
print(f"Output: {OUTPUT_PATH}")