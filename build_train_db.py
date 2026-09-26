import csv
import os
import re
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    "train_candidate_index.db"
)

S2_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source3.tsv"
)


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("Creating training candidate database...")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

conn.execute("""
CREATE TABLE records (
    entity_id TEXT PRIMARY KEY,
    country TEXT,
    name_norm TEXT,
    address_norm TEXT
)
""")

conn.execute("""
CREATE INDEX idx_name
ON records(country, name_norm)
""")

conn.execute("""
CREATE INDEX idx_address
ON records(country, address_norm)
""")


def load_file(path, label):

    print(f"Loading {label}...")

    count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        batch = []

        for row in reader:

            entity_id = row["entity_id"]

            country = row["country"] or ""

            name_norm = normalize(
                row["business_name"]
            )

            address_norm = normalize(
                row["business_address"]
            )

            batch.append(
                (
                    entity_id,
                    country,
                    name_norm,
                    address_norm
                )
            )

            count += 1

            if len(batch) >= 10000:

                conn.executemany(
                    """
                    INSERT INTO records
                    VALUES (?, ?, ?, ?)
                    """,
                    batch
                )

                batch.clear()

            if count % 100000 == 0:

                print(
                    f"{label}: {count:,}"
                )

        if batch:

            conn.executemany(
                """
                INSERT INTO records
                VALUES (?, ?, ?, ?)
                """,
                batch
            )

    print(
        f"{label} complete: {count:,}"
    )

    return count


s2_count = load_file(
    S2_PATH,
    "Source 2"
)

s3_count = load_file(
    S3_PATH,
    "Source 3"
)

print("Creating indexes...")

conn.commit()

conn.execute(
    "REINDEX"
)

conn.commit()

conn.close()

print()
print("Training candidate database complete.")
print(
    f"Source 2 records: {s2_count:,}"
)
print(
    f"Source 3 records: {s3_count:,}"
)
print(
    f"Total records: "
    f"{s2_count + s3_count:,}"
)
print(
    f"Database: {DB_PATH}"
)