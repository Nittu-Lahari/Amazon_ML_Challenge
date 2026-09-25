import pandas as pd

file_path = "dataset/train/train_ground_truth.tsv"

df = pd.read_csv(file_path, sep="\t")

print("=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)

# Total Source 1 entities
total = len(df)

# Empty match lists = no match
no_match = df["matched_entity_ids"].isna().sum()

# Convert missing values to empty strings
matches = df["matched_entity_ids"].fillna("")

# Count number of matched entities for every Source 1 record
match_count = matches.apply(
    lambda x: 0 if x == "" else len(x.split(","))
)

print("\nTotal Source 1 entities:", total)

print("\nNumber of matches per Source 1 entity:")
print("No matches :", (match_count == 0).sum())
print("1 match    :", (match_count == 1).sum())
print("2 matches  :", (match_count == 2).sum())
print("3 matches  :", (match_count == 3).sum())
print("4 matches  :", (match_count == 4).sum())
print("5 matches  :", (match_count == 5).sum())
print("6+ matches :", (match_count >= 6).sum())

# Count S2 and S3 matches
def count_s2(x):
    if x == "":
        return 0
    return sum(1 for item in x.split(",") if item.startswith("S2-"))

def count_s3(x):
    if x == "":
        return 0
    return sum(1 for item in x.split(",") if item.startswith("S3-"))

s2_counts = matches.apply(count_s2)
s3_counts = matches.apply(count_s3)

print("\nMatch source distribution:")

print("Entities having S2 match:",
      (s2_counts > 0).sum())

print("Entities having S3 match:",
      (s3_counts > 0).sum())

print("Entities having BOTH S2 and S3 matches:",
      ((s2_counts > 0) & (s3_counts > 0)).sum())

print("\nTotal S2 matches:", s2_counts.sum())
print("Total S3 matches:", s3_counts.sum())

print("\nMaximum matches for one Source 1 entity:",
      match_count.max())

print("\nAverage matches per Source 1 entity:",
      match_count.mean())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)