import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Amazon Review Issue Clustering", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/reviews_clustered.csv")
    df["cluster_label"] = df["cluster_label"].fillna("Unclustered")
    return df

df = load_data()

clustered   = df[df["cluster_id"] != -1]
noise       = df[df["cluster_id"] == -1]
n_clusters  = clustered["cluster_id"].nunique()
n_noise     = len(noise)
n_total     = len(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Amazon Review Issue Clustering")
    st.caption("NLP Portfolio Project — Issue Detection from Unstructured Text")
    st.divider()
    st.metric("Reviews Analyzed", f"{n_total:,}")
    st.metric("Clusters Found",   f"{n_clusters}")
    st.metric("Unclustered",      f"{n_noise:,}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Cluster Overview", "🗺️ Issue Map", "🔍 Explore a Cluster"])

# ── Tab 1: Bar chart ───────────────────────────────────────────────────────────
with tab1:
    st.header("Top Issue Clusters in Amazon Product Reviews")

    summary = (
        clustered
        .groupby("cluster_label")
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=True)
    )

    fig1 = px.bar(
        summary,
        x="review_count",
        y="cluster_label",
        orientation="h",
        color="review_count",
        color_continuous_scale="Blues",
        labels={"review_count": "Number of Reviews", "cluster_label": "Issue Cluster"},
        title="Top Issue Clusters in Amazon Product Reviews",
    )
    fig1.update_layout(
        height=700,
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title="Number of Reviews",
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Tab 2: UMAP scatter ────────────────────────────────────────────────────────
with tab2:
    st.header("Semantic Map of Customer Complaints")

    plot_df = df.copy()
    plot_df["display_label"] = plot_df["cluster_label"].where(
        plot_df["cluster_id"] != -1, "Unclustered"
    )
    plot_df["hover_text"] = plot_df["review_text"].str[:100]

    # Build a color map: Unclustered = grey, rest from Plotly palette
    labels_ordered = sorted(
        plot_df[plot_df["cluster_id"] != -1]["display_label"].unique()
    )
    color_map = {"Unclustered": "#c0c0c0"}

    fig2 = px.scatter(
        plot_df.sort_values("cluster_id"),
        x="umap_x",
        y="umap_y",
        color="display_label",
        color_discrete_map=color_map,
        hover_data={"display_label": True, "hover_text": True,
                    "umap_x": False, "umap_y": False},
        labels={"display_label": "Cluster", "hover_text": "Review"},
        title="Semantic Map of Customer Complaints",
        opacity=0.75,
    )
    fig2.update_traces(marker=dict(size=5))
    fig2.update_layout(
        height=650,
        legend=dict(title="Cluster", itemsizing="constant"),
        xaxis_title=None,
        yaxis_title=None,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Cluster explorer ────────────────────────────────────────────────────
with tab3:
    st.header("Explore a Cluster")

    label_options = (
        clustered
        .groupby("cluster_label")
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=False)["cluster_label"]
        .tolist()
    )

    selected_label = st.selectbox("Select an issue cluster:", label_options)

    if selected_label:
        subset = clustered[clustered["cluster_label"] == selected_label]
        count  = len(subset)

        st.subheader(selected_label)
        st.markdown(f"**{count} reviews in this cluster**")
        st.divider()

        sample_cols = ["rating", "product_name", "review_text"]
        sample = subset[sample_cols].head(10).reset_index(drop=True)
        sample.index += 1
        st.dataframe(sample, use_container_width=True)

        st.divider()
        st.info(
            f"💡 **Business Insight:** {selected_label} affects {count} reviews. "
            f"A product team should prioritize this in the next sprint."
        )
