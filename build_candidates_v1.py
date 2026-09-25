import csv
import re
import sqlite3
import os


TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

DB_FILE = "candidate_index.db"

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


def normalize(value):
    if not value:
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def create_database():

    if os.path.exists(DB_FILE):
        print("Existing database found. Reusing it.")
        return

    print("Creating candidate index...")

    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            name_norm TEXT,
            address_norm TEXT
        )
    """)

    print("Reading Source 2...")

    insert_sql = """
        INSERT INTO records
        (entity_id, country, name_norm, address_norm)
        VALUES (?, ?, ?, ?)
    """

    with open(
        S2_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        batch = []

        for row in reader:

            batch.append((
                row["entity_id"],
                row["country"],
                normalize(row["business_name"]),
                normalize(row["business_address"])
            ))

            if len(batch) >= 10000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()

    print("Source 2 indexed.")

    print("Reading Source 3...")

    with open(
        S3_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        batch = []

        for row in reader:

            batch.append((
                row["entity_id"],
                row["country"],
                normalize(row["business_name"]),
                normalize(row["business_address"])
            ))

            if len(batch) >= 10000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()

    print("Source 3 indexed.")

    print("Creating indexes...")

    conn.execute("""
        CREATE INDEX idx_name
        ON records(country, name_norm)
    """)

    conn.execute("""
        CREATE INDEX idx_address
        ON records(country, address_norm)
    """)

    conn.commit()
    conn.close()

    print("Candidate index created successfully.")


def generate_candidates():

    print("\nGenerating candidates...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_FILE)

    name_query = """
        SELECT entity_id
        FROM records
        WHERE country = ?
        AND name_norm = ?
        AND name_norm != ''
    """

    address_query = """
        SELECT entity_id
        FROM records
        WHERE country = ?
        AND address_norm = ?
        AND address_norm != ''
    """

    with open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as out:

        writer = csv.writer(
            out,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])

        with open(
            S1_FILE,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as f:

            reader = csv.DictReader(
                f,
                delimiter="\t"
            )

            count = 0

            for row in reader:

                s1_id = row["entity_id"]
                country = row["country"]

                name_norm = normalize(
                    row["business_name"]
                )

                address_norm = normalize(
                    row["business_address"]
                )

                candidates = set()

                # Name candidates
                if name_norm:
                    rows = conn.execute(
                        name_query,
                        (country, name_norm)
                    )

                    for result in rows:
                        candidates.add(result[0])

                # Address candidates
                if address_norm:
                    rows = conn.execute(
                        address_query,
                        (country, address_norm)
                    )

                    for result in rows:
                        candidates.add(result[0])

                # Safety: never allow S1 IDs
                candidates = {
                    x for x in candidates
                    if x.startswith("S2-")
                    or x.startswith("S3-")
                }

                writer.writerow([
                    s1_id,
                    ",".join(sorted(candidates))
                ])

                count += 1

                if count % 10000 == 0:
                    print(
                        f"Processed Source 1 records: {count:,}"
                    )

    conn.close()

    print("\nCandidate generation completed.")
    print("Output:", CANDIDATE_FILE)


if __name__ == "__main__":

    create_database()
    generate_candidates()
    