---
title: Amazon Review Issue Clustering
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app/streamlit_app.py
pinned: false
---

# Amazon Review Issue Clustering
### NLP Portfolio Project — Extracting Actionable Insights from Unstructured Text

## 🔍 What This Project Does
This system automatically discovers recurring issue patterns in Amazon
product reviews using semantic embeddings and unsupervised clustering —
without any predefined categories or labels.

A product team can use this dashboard to answer:
"What are our customers actually complaining about, and how many reviews
does each issue affect?"

## 🚀 Live Demo
[Link to Hugging Face Spaces — add after deployment]

## 📊 Key Results
| Metric | Value |
|--------|-------|
| Reviews analyzed | 1,156 negative reviews (1-2 star) |
| Clusters discovered | 34 distinct issue groups |
| Silhouette Score | 0.46 (reasonable — expected for overlapping complaint language) |
| Battery-related issues | 45.9% of all negative reviews |
| Device failure issues | 22.1% of all negative reviews |

**Top actionable insight:** Nearly half of all 1-2 star reviews are about
battery longevity. The AmazonBasics battery line is the primary driver of
negative sentiment and should be the first product team priority.

## 🏗️ Architecture
```
Raw CSV (28,332 reviews)
    → Cleaning and filtering (17,125 retained)
    → Negative review extraction (1,156 reviews, 1-2 star)
    → OpenAI text-embedding-3-small (1,536-dim vectors)
    → UMAP dimensionality reduction (1,536D → 10D for clustering, 2D for viz)
    → HDBSCAN clustering (34 clusters found)
    → GPT-4o-mini cluster labeling (auto-generated human-readable names)
    → Streamlit dashboard (3-tab interactive app)
```

## 💡 Business Workflow Integration
This system fits into a product team's weekly workflow as follows:
- Monday morning: Dashboard refreshes with new reviews from the past week
- Product manager opens Tab 1 to see if any new issue cluster has grown
- Engineering lead opens Tab 3 to drill into clusters before sprint planning
- QA team uses Tab 2 to spot semantically related issues sharing a root cause

Without this system, identifying these patterns would require manually
reading hundreds of reviews. This system surfaces them in seconds.

## 🛠️ Tech Stack
| Component | Tool | Why |
|-----------|------|-----|
| Embeddings | OpenAI text-embedding-3-small | Semantic understanding beyond keyword matching |
| Clustering | HDBSCAN | Finds natural groupings without preset cluster count |
| Dimensionality Reduction | UMAP | Preserves local and global structure better than PCA |
| Cluster Labeling | GPT-4o-mini | Auto-generates human-readable names from sample reviews |
| Visualization | Plotly | Interactive hover tooltips and color-coded scatter plots |
| App | Streamlit | Accessible to non-engineers, no frontend coding required |

## 📐 Design Decisions and Tradeoffs

**Why HDBSCAN over K-Means?**
K-Means requires specifying the number of clusters upfront. With complaint
data we do not know how many issue types exist — HDBSCAN discovers that
automatically and handles noise points gracefully.

**Why embed only negative reviews?**
Clustering all 17,125 reviews would mix praise and complaints, producing
meaningless clusters. Filtering to 1-2 star reviews focuses the embedding
space on the signal we care about.

**Why text-embedding-3-small over Sentence Transformers?**
OpenAI embeddings outperform most open-source alternatives on short noisy
text like product reviews. The cost for 1,156 reviews is under $0.01.

**Silhouette score of 0.46 — is that good?**
For real-world complaint data, yes. Many battery-related clusters are
semantically similar by nature, which naturally lowers the score. The
qualitative review confirms the top clusters are coherent and actionable.

## 📁 Project Structure
```
amazon-review-clustering/
├── data/
│   ├── processed/          <- cleaned CSVs and clustered output
│   └── embeddings/         <- .npy files (gitignored, generate locally)
├── src/
│   ├── preprocess.py       <- data inspection
│   ├── clean.py            <- cleaning pipeline
│   ├── embed.py            <- OpenAI embedding generation
│   ├── cluster.py          <- UMAP + HDBSCAN clustering
│   ├── label_clusters.py   <- GPT-4o-mini cluster labeling
│   └── evaluate.py         <- silhouette score + qualitative evaluation
├── app/
│   └── streamlit_app.py    <- 3-tab interactive dashboard
├── requirements.txt
└── README.md
```

## ⚙️ Setup and Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/amazon-review-clustering
cd amazon-review-clustering
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
python src/clean.py
python src/embed.py
python src/cluster.py
python src/label_clusters.py
python src/evaluate.py
streamlit run app/streamlit_app.py
```

## 📈 Evaluation
- Silhouette Score: 0.46 — reasonable separation given overlapping complaint language
- Top coherent clusters: Battery drain on Roku/Fire remotes, Battery leakage, Brand comparisons
- Mixed clusters: General early product failure — spans multiple product types as expected
- 31.9% noise: Acceptable — reflects genuinely ambiguous reviews that do not fit one issue pattern

---
Built as part of a data science portfolio following Aishwarya Srinivasan's
"5 Data Science Projects that will Get You into Big Techs" framework.
