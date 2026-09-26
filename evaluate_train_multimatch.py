import csv
import os
import re
import sqlite3
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

S1_PATH = os.path.join(
    BASE_DIR, "dataset", "train", "train_source1.tsv"
)

GT_PATH = os.path.join(
    BASE_DIR, "dataset", "train", "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    BASE_DIR, "output", "train_candidate_pairs.tsv"
)

DB_PATH = os.path.join(
    BASE_DIR, "train_candidate_index.db"
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


print("Loading training S1...")

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

print(f"S1 loaded: {len(s1):,}")


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
    f"Ground truth loaded: "
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
    f"Candidate lists loaded: "
    f"{len(candidates):,}"
)


print("Loading candidate records...")

needed_ids = set()

for ids in candidates.values():
    needed_ids.update(ids)

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
    f"Candidate records loaded: "
    f"{len(record_map):,}"
)


print("Scoring candidates...")

# Store feature tuples per S1.
# This avoids recalculating similarities for every
# weight/threshold experiment.

scored = {}

processed = 0

for s1_id, true_ids in ground_truth.items():

    if not true_ids:
        continue

    s1_row = s1.get(s1_id)

    if s1_row is None:
        continue

    country, s1_name, s1_address = s1_row

    rows = []

    for candidate_id in candidates.get(s1_id, []):

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

        address_seq = similarity(
            s1_address,
            c_address
        )

        name_jac = token_jaccard(
            s1_name,
            c_name
        )

        address_jac = token_jaccard(
            s1_address,
            c_address
        )

        rows.append(
            (
                candidate_id,
                name_seq,
                address_seq,
                name_jac,
                address_jac
            )
        )

    if rows:
        scored[s1_id] = rows

    processed += 1

    if processed % 100000 == 0:
        print(
            f"Processed S1: {processed:,}"
        )


print(
    f"Scored S1 records: "
    f"{len(scored):,}"
)


def calculate_f05(
    predicted,
    actual
):

    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    denominator = (
        0.25 * precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / denominator
    )


def evaluate(
    wnseq,
    waseq,
    wnjac,
    wajac,
    best_threshold,
    extra_threshold,
    margin
):

    total_f05 = 0.0
    evaluated = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1_id, actual in ground_truth.items():

        if not actual:
            # Correct empty prediction gets 1.0.
            total_f05 += 1.0
            evaluated += 1
            continue

        rows = scored.get(s1_id)

        if not rows:
            evaluated += 1
            continue

        scored_rows = []

        for (
            candidate_id,
            name_seq,
            address_seq,
            name_jac,
            address_jac
        ) in rows:

            score = (
                wnseq * name_seq
                + waseq * address_seq
                + wnjac * name_jac
                + wajac * address_jac
            )

            scored_rows.append(
                (
                    candidate_id,
                    score
                )
            )

        scored_rows.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_id, best_score = scored_rows[0]

        predicted = set()

        if best_score >= best_threshold:

            predicted.add(best_id)

            for candidate_id, score in scored_rows[1:]:

                if (
                    score >= extra_threshold
                    and
                    best_score - score <= margin
                ):
                    predicted.add(candidate_id)

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_f05 += calculate_f05(
            predicted,
            actual
        )

        evaluated += 1

    macro_f05 = (
        total_f05 / evaluated
        if evaluated
        else 0.0
    )

    return (
        macro_f05,
        total_tp,
        total_fp,
        total_fn
    )


# ---------------------------------------------------------
# Experiments
# ---------------------------------------------------------

strategies = [
    (
        "V2_STYLE",
        0.60, 0.40, 0.0, 0.0
    ),
    (
        "ALL_EQUAL",
        0.25, 0.25, 0.25, 0.25
    ),
    (
        "ADDRESS_SEQ",
        0.10, 0.70, 0.05, 0.15
    ),
    (
        "ADDRESS_JAC",
        0.10, 0.15, 0.05, 0.70
    ),
    (
        "ADDRESS_HEAVY",
        0.10, 0.55, 0.10, 0.25
    ),
    (
        "ADDRESS_BALANCED",
        0.15, 0.45, 0.10, 0.30
    ),
    (
        "NAME_SEQ_ADDRESS_JAC",
        0.20, 0.10, 0.10, 0.60
    )
]


best_results = []


print()
print("Evaluating multi-match F0.5...")
print()


for (
    name,
    wnseq,
    waseq,
    wnjac,
    wajac
) in strategies:

    for best_threshold in [
        0.70,
        0.75,
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
        0.92,
        0.94
    ]:

        for extra_threshold in [
            0.88,
            0.90,
            0.92,
            0.94
        ]:

            result = evaluate(
                wnseq,
                waseq,
                wnjac,
                wajac,
                best_threshold,
                extra_threshold,
                0.03
            )

            macro_f05, tp, fp, fn = result

            best_results.append(
                (
                    macro_f05,
                    name,
                    best_threshold,
                    extra_threshold,
                    tp,
                    fp,
                    fn
                )
            )


best_results.sort(
    key=lambda x: x[0],
    reverse=True
)


print()
print("TOP MULTI-MATCH STRATEGIES")
print("==========================")

for result in best_results[:25]:

    (
        f05,
        name,
        best_threshold,
        extra_threshold,
        tp,
        fp,
        fn
    ) = result

    print(
        f"{name:24s} "
        f"best={best_threshold:.2f} "
        f"extra={extra_threshold:.2f} "
        f"F0.5={f05:.4f} "
        f"TP={tp:,} "
        f"FP={fp:,} "
        f"FN={fn:,}"
    )


print()
print("Evaluation complete.")
