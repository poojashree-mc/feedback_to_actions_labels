"""
dashboard.py
Streamlit dashboard for visualising processed feedback and action recommendations.

Usage:
    streamlit run src/dashboard.py
"""

import os
import json

import pandas as pd
import streamlit as st

OUTPUT_CSV_PATH = os.path.join("data", "output", "output_feedback_actions.csv")


# =========  DATA  =========

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{path}' not found.\n"
            "Run `python src/processor.py` first to generate it."
        )

    df = pd.read_csv(path)

    expected_cols = {"label", "text", "sentiment", "themes_json", "actions_json"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    def _parse(x):
        try:
            return json.loads(x) if isinstance(x, str) and x.strip() else []
        except Exception:
            return []

    df["themes"] = df["themes_json"].apply(_parse)
    df["actions"] = df["actions_json"].apply(_parse)
    df["theme_names"] = df["themes"].apply(
        lambda ts: [t.get("name") for t in ts if isinstance(t, dict) and "name" in t]
    )
    return df


# =========  FILTERS  =========

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    sentiments = sorted(df["sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

    all_themes = sorted({t for row in df["theme_names"] for t in row})
    selected_themes = st.sidebar.multiselect("Themes", all_themes, default=all_themes)

    search_query = st.sidebar.text_input("Search review text", value="")

    filtered = df.copy()

    if selected_sentiments:
        filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]

    if selected_themes and len(selected_themes) != len(all_themes):
        filtered = filtered[
            filtered["theme_names"].apply(lambda ts: any(t in selected_themes for t in ts))
        ]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[filtered["text"].str.lower().str.contains(q, na=False)]

    return filtered


# =========  MAIN  =========

def main():
    st.set_page_config(page_title="Feedback → Action Dashboard", layout="wide")
    st.title("📊 Feedback → Action Dashboard")
    st.caption("Rule-based sentiment analysis, theme detection & action recommendations")

    try:
        df = load_data(OUTPUT_CSV_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    filtered_df = apply_filters(df)

    # --- Top-level metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Filtered Reviews", len(filtered_df))

    sentiment_counts = filtered_df["sentiment"].value_counts()
    top_sent = sentiment_counts.index[0] if not sentiment_counts.empty else "N/A"
    col3.metric("Most Common Sentiment", top_sent)

    st.markdown("---")

    # --- Review list ---
    st.subheader("Reviews & Recommendations")

    if filtered_df.empty:
        st.info("No reviews match the current filters.")
        return

    max_rows = st.slider("Max reviews to display", min_value=1, max_value=50, value=10)

    for _, row in filtered_df.head(max_rows).iterrows():
        sentiment = row["sentiment"]
        text = row["text"]
        preview = text[:120].replace("\n", " ") + ("..." if len(text) > 120 else "")

        with st.expander(f"[{sentiment.upper()}] {preview}"):
            st.write("**Original Label:**", row["label"])
            st.write("**Full Review:**")
            st.write(text)

            st.write("**Detected Themes:**")
            if row["themes"]:
                for t in row["themes"]:
                    st.write(f"- `{t.get('name', 'unknown')}` (confidence: {t.get('confidence', 0):.2f})")
            else:
                st.write("_No themes detected_")

            st.write("**Recommended Actions:**")
            if row["actions"]:
                for a in row["actions"]:
                    st.write(
                        f"- **{a.get('title')}** "
                        f"(ID: `{a.get('action_id')}`, "
                        f"Team: `{a.get('team')}`, "
                        f"Priority: {a.get('priority')})"
                    )
                    st.write(f"  - Reason: {a.get('reason')}")
            else:
                st.write("_No actions recommended_")

    st.markdown("---")
    st.caption("Data source: local CSV processed with rule-based pipeline.")


if __name__ == "__main__":
    main()
