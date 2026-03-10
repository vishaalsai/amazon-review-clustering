import pandas as pd

CSV_PATH = "data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"

df = pd.read_csv(CSV_PATH, usecols=["reviews.text", "reviews.title", "reviews.rating", "name"])

df = df.rename(columns={
    "reviews.text":   "review_text",
    "reviews.title":  "review_title",
    "reviews.rating": "rating",
    "name":           "product_name",
})

before = len(df)

df = df[df["review_text"].notna()]
df = df[df["review_text"].str.strip() != ""]
df = df[df["review_text"].str.len() >= 20]
df = df.drop_duplicates(subset="review_text")

df["is_negative"] = df["rating"] <= 2

after       = len(df)
n_negative  = df["is_negative"].sum()
n_positive  = (~df["is_negative"]).sum()

print("=== CLEANING SUMMARY ===")
print(f"  Rows before cleaning : {before:,}")
print(f"  Rows after cleaning  : {after:,}")
print(f"  Dropped              : {before - after:,}")
print()
print(f"  Negative reviews (1-2 stars) : {n_negative:,}")
print(f"  Positive reviews (3-5 stars) : {n_positive:,}")

df.to_csv("data/processed/reviews_clean.csv", index=False)
df[df["is_negative"]].to_csv("data/processed/reviews_negative.csv", index=False)

print()
print("  Saved -> data/processed/reviews_clean.csv")
print("  Saved -> data/processed/reviews_negative.csv")
