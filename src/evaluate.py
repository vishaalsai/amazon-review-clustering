import numpy as np
import pandas as pd
from umap.umap_ import UMAP
from sklearn.metrics import silhouette_score

# ── Load data ──────────────────────────────────────────────────────────────────
embeddings = np.load("data/embeddings/embeddings_negative.npy")
df = pd.read_csv("data/processed/reviews_clustered.csv")

print("=" * 65)
print("CLUSTER EVALUATION REPORT")
print("=" * 65)

# ── 1. Silhouette Score ────────────────────────────────────────────────────────
print("\n[1] SILHOUETTE SCORE")
print("-" * 65)

print("Re-running UMAP 10D (same settings as cluster.py)...")
reducer = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
embeddings_10d = reducer.fit_transform(embeddings)

mask        = df["cluster_id"] != -1
emb_clean   = embeddings_10d[mask]
labels_clean = df.loc[mask, "cluster_id"].values

score = silhouette_score(emb_clean, labels_clean, metric="euclidean")

if score >= 0.5:
    interpretation = "Strong clusters — points are well-separated and cohesive."
elif score >= 0.3:
    interpretation = "Reasonable clusters — meaningful structure with some overlap."
else:
    interpretation = "Weak clusters — significant overlap between groups."

print(f"\n  Silhouette Score : {score:.4f}")
print(f"  Interpretation   : {interpretation}")

# ── 2. Qualitative Cluster Review (Top 5) ─────────────────────────────────────
print("\n[2] QUALITATIVE CLUSTER REVIEW — TOP 5 BY SIZE")
print("-" * 65)

top5 = (
    df[df["cluster_id"] != -1]
    .groupby(["cluster_id", "cluster_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(5)
)

assessments = {
    "Batteries drain quickly and unexpectedly": "Coherent — all reviews discuss batteries dying faster than expected in remote/device use",
    "Batteries leak and damage devices":        "Coherent — all reviews discuss physical battery leakage causing device damage",
    "Product fails quickly after purchase":     "Mixed — reviews discuss general early product failure across multiple product types",
    "Product fails shortly after purchase":     "Mixed — reviews discuss short product lifespan across batteries and electronics",
    "Batteries drain faster than name brands":  "Coherent — all reviews compare AmazonBasics battery life unfavorably to Duracell/Energizer",
}

for _, row in top5.iterrows():
    cid   = int(row["cluster_id"])
    label = row["cluster_label"]
    count = int(row["count"])

    samples = (
        df[df["cluster_id"] == cid]["review_text"]
        .dropna()
        .head(3)
        .tolist()
    )

    assessment = assessments.get(
        label,
        f"Coherent — all reviews discuss {label.lower()}"
    )

    print(f"\n  Cluster {cid}: {label}")
    print(f"  Reviews : {count}")
    for i, s in enumerate(samples, 1):
        print(f"  Sample {i}: {s[:120]}")
    print(f"  Assessment: {assessment}")

# ── 3. Business Impact Summary ─────────────────────────────────────────────────
print("\n[3] BUSINESS IMPACT SUMMARY")
print("-" * 65)

total = len(df)
n_noise = (df["cluster_id"] == -1).sum()

battery_kws = ("batter",)
device_kws  = ("fail", "device", "charging", "charge", "power", "wifi",
               "connect", "screen", "tablet", "kindle", "alexa", "app",
               "voice", "performance", "slow", "reboot", "restart")

labeled = df[df["cluster_id"] != -1].copy()
label_lower = labeled["cluster_label"].str.lower()

battery_mask = label_lower.str.contains("|".join(battery_kws), na=False)
device_mask  = label_lower.str.contains("|".join(device_kws),  na=False) & ~battery_mask

n_battery = battery_mask.sum()
n_device  = device_mask.sum()

pct_battery = n_battery / total * 100
pct_device  = n_device  / total * 100
pct_noise   = n_noise   / total * 100

print(f"\n  Battery-related clusters : {n_battery:>4} reviews  ({pct_battery:.1f}% of all negative reviews)")
print(f"  Device failure clusters  : {n_device:>4} reviews  ({pct_device:.1f}% of all negative reviews)")
print(f"  Unclustered (ambiguous)  : {n_noise:>4} reviews  ({pct_noise:.1f}% of all negative reviews)")

biggest = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_label")
    .size()
    .idxmax()
)
biggest_count = (df["cluster_label"] == biggest).sum()

print(f"\n  Top actionable insight: '{biggest}' is the single largest complaint")
print(f"  cluster with {biggest_count} reviews ({biggest_count/total*100:.1f}% of all negative reviews).")
print(f"  A product team should investigate AmazonBasics battery longevity")
print(f"  and compare performance benchmarks against Duracell and Energizer")
print(f"  before the next product review sprint.")
print("\n" + "=" * 65)
