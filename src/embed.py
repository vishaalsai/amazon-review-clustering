import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100

df = pd.read_csv("data/processed/reviews_negative.csv")

df["review_title"] = df["review_title"].fillna("")
df["combined_text"] = df["review_title"].str.strip() + ". " + df["review_text"].str.strip()
df = df.reset_index(drop=True)
df.insert(0, "index", df.index)

texts = df["combined_text"].tolist()

all_embeddings = []
batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

for batch in tqdm(batches, desc="Embedding batches"):
    response = client.embeddings.create(input=batch, model=EMBED_MODEL)
    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)

embeddings = np.array(all_embeddings)

np.save("data/embeddings/embeddings_negative.npy", embeddings)

df[["index", "product_name", "rating", "review_text"]].to_csv(
    "data/processed/reviews_negative_indexed.csv", index=False
)

print(f"\nDone.")
print(f"  Total embeddings saved : {embeddings.shape[0]:,}")
print(f"  Embedding array shape  : {embeddings.shape}")
print(f"  Saved -> data/embeddings/embeddings_negative.npy")
print(f"  Saved -> data/processed/reviews_negative_indexed.csv")
