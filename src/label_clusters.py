import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SAMPLES_PER_CLUSTER = 8
MODEL = "gpt-4o-mini"

df = pd.read_csv("data/processed/reviews_clustered.csv")

clusters = sorted(df[df["cluster_id"] != -1]["cluster_id"].unique())
print(f"Labeling {len(clusters)} clusters...\n")

labels = {}

for cid in tqdm(clusters, desc="Labeling clusters"):
    samples = (
        df[df["cluster_id"] == cid]["review_text"]
        .dropna()
        .head(SAMPLES_PER_CLUSTER)
        .tolist()
    )
    reviews_block = "\n".join(f"- {r.strip()}" for r in samples)

    prompt = (
        f"You are analyzing Amazon product complaints. Here are {len(samples)} customer "
        f"reviews that all belong to the same issue cluster:\n\n"
        f"{reviews_block}\n\n"
        f"Give this cluster a short, specific label (4-7 words max) that describes "
        f"the core product issue. Be specific — not 'product quality issue' but "
        f"something like 'Battery dies within first month' or "
        f"'WiFi disconnects randomly during use'.\n\n"
        f"Respond with ONLY the label, nothing else."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    labels[cid] = response.choices[0].message.content.strip()

df["cluster_label"] = df["cluster_id"].map(labels).fillna("Noise")
df.to_csv("data/processed/reviews_clustered.csv", index=False)

print("\n=== CLUSTER LABELS ===")
print(f"{'ID':>4}  {'Count':>6}  Label")
print("-" * 70)

summary = (
    df[df["cluster_id"] != -1]
    .groupby(["cluster_id", "cluster_label"])
    .size()
    .reset_index(name="review_count")
    .sort_values("review_count", ascending=False)
)

for _, row in summary.iterrows():
    print(f"{int(row['cluster_id']):>4}  {int(row['review_count']):>6}  {row['cluster_label']}")

print(f"\nSaved -> data/processed/reviews_clustered.csv")
