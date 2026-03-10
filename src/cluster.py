import numpy as np
import pandas as pd
from umap.umap_ import UMAP
import hdbscan

embeddings = np.load("data/embeddings/embeddings_negative.npy")
df = pd.read_csv("data/processed/reviews_negative_indexed.csv")

print(f"Loaded {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")

# --- UMAP: 1536 -> 10D for clustering ---
print("\nRunning UMAP 10D reduction...")
reducer_10d = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
embeddings_10d = reducer_10d.fit_transform(embeddings)

# --- UMAP: 1536 -> 2D for visualization ---
print("Running UMAP 2D reduction...")
reducer_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
embeddings_2d = reducer_2d.fit_transform(embeddings)

# --- HDBSCAN on 10D ---
print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, cluster_selection_method="leaf")
labels = clusterer.fit_predict(embeddings_10d)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()

print(f"\nClusters found : {n_clusters}")
print(f"Noise points   : {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")

df["cluster_id"] = labels
df["umap_x"]     = embeddings_2d[:, 0]
df["umap_y"]     = embeddings_2d[:, 1]

# --- Cluster summary ---
print("\n=== CLUSTER SUMMARY ===")
print(f"{'Cluster':>8}  {'Count':>6}  {'Sample Reviews'}")
print("-" * 90)

for cid in sorted(df["cluster_id"].unique()):
    subset  = df[df["cluster_id"] == cid]
    count   = len(subset)
    samples = subset["review_text"].dropna().head(2).tolist()
    label   = f"{cid:>8}" if cid != -1 else "   NOISE"
    print(f"{label}  {count:>6}")
    for s in samples:
        print(f"            > {str(s)[:80]}")
    print()

df.to_csv("data/processed/reviews_clustered.csv", index=False)
print("Saved -> data/processed/reviews_clustered.csv")
