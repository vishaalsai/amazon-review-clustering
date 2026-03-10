import pandas as pd

CSV_PATH = "data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"

df = pd.read_csv(CSV_PATH)

print("=== COLUMN NAMES ===")
for col in df.columns:
    print(f"  {col}")

print(f"\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print(f"\n=== TOTAL ROWS ===")
print(f"  {len(df):,}")

print(f"\n=== RATING/STARS COLUMN VALUE COUNTS ===")
rating_cols = [col for col in df.columns if any(kw in col.lower() for kw in ("rating", "star", "score"))]
if rating_cols:
    for col in rating_cols:
        print(f"\n  Column: '{col}'")
        print(df[col].value_counts().sort_index().to_string())
else:
    print("  No rating/stars columns detected.")
