import csv
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "train_candidate_index.db")

S1_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source1.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "output",
    "train_candidate_pairs.tsv"
)


print("Opening candidate database...")

conn = sqlite3.connect(DB_PATH)

name_cur = conn.cursor()
address_cur = conn.cursor()


print("Loading training Source-1 records...")

s1_rows = []

with open(
    S1_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        s1_rows.append(
            (
                row["entity_id"],
                row["country"] or "",
                row["business_name"] or "",
                row["business_address"] or ""
            )
        )

print(
    f"Loaded S1 records: "
    f"{len(s1_rows):,}"
)


def normalize(text):

    if not isinstance(text, str):
        return ""

    import re

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


print("Building training candidate pairs...")

total = 0
rows_with_candidates = 0
total_candidates = 0
max_candidates = 0

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as out_file:

    writer = csv.writer(
        out_file,
        delimiter="\t"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    for (
        s1_id,
        country,
        business_name,
        business_address
    ) in s1_rows:

        name_norm = normalize(
            business_name
        )

        address_norm = normalize(
            business_address
        )

        candidate_ids = set()

        # Exact normalized name candidates.
        if name_norm:

            rows = name_cur.execute(
                """
                SELECT entity_id
                FROM records
                WHERE country = ?
                  AND name_norm = ?
                LIMIT 21
                """,
                (
                    country,
                    name_norm
                )
            )

            for row in rows:
                candidate_ids.add(row[0])

        # Exact normalized address candidates.
        if address_norm:

            rows = address_cur.execute(
                """
                SELECT entity_id
                FROM records
                WHERE country = ?
                  AND address_norm = ?
                LIMIT 11
                """,
                (
                    country,
                    address_norm
                )
            )

            for row in rows:
                candidate_ids.add(row[0])

        candidate_list = sorted(
            candidate_ids
        )

        if candidate_list:
            rows_with_candidates += 1

        total_candidates += len(
            candidate_list
        )

        max_candidates = max(
            max_candidates,
            len(candidate_list)
        )

        writer.writerow([
            s1_id,
            ",".join(candidate_list)
        ])

        total += 1

        if total % 100000 == 0:

            print(
                f"Processed: {total:,} | "
                f"Rows with candidates: "
                f"{rows_with_candidates:,} | "
                f"Candidates: "
                f"{total_candidates:,}"
            )


conn.close()

print()
print("Training candidate generation complete.")
print(
    f"S1 rows: {total:,}"
)
print(
    f"Rows with candidates: "
    f"{rows_with_candidates:,}"
)
print(
    f"Total candidates: "
    f"{total_candidates:,}"
)
print(
    f"Max candidates per S1: "
    f"{max_candidates:,}"
)
print(
    f"Output: {OUTPUT_PATH}"
)