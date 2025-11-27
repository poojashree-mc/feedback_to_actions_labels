import os
import json
import pandas as pd
import streamlit as st

# Path to the processed CSV from the previous script
OUTPUT_CSV_PATH = "output_feedback_actions.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File '{path}' not found.\n"
            f"Run your processing script first to generate it."
        )

    df = pd.read_csv(path)

    # Ensure expected columns exist
    expected_cols = {"label", "text", "sentiment", "themes_json", "actions_json"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Parse JSON columns into Python objects
    def parse_json_safe(x):
        try:
            return json.loads(x) if isinstance(x, str) and x.strip() else []
        except Exception:
            return []

    df["themes"] = df["themes_json"].apply(parse_json_safe)
    df["actions"] = df["actions_json"].apply(parse_json_safe)

    # Extract simple theme names for filtering
    df["theme_names"] = df["themes"].apply(
        lambda ts: [t.get("name") for t in ts if isinstance(t, dict) and "name" in t]
    )

    return df


def main():
    st.set_page_config(
        page_title="Feedback → Action Dashboard",
        layout="wide",
    )

    st.title("📊 Feedback → Action Dashboard")
    st.caption("Local rule-based sentiment, theme detection, and action recommendations")

    # Load data
    try:
        df = load_data(OUTPUT_CSV_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Sentiment filter
    sentiments = sorted(df["sentiment"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=sentiments,
        default=sentiments,
    )

    # Theme filter
    # Flatten theme names list
    all_themes = sorted(
        {t for row in df["theme_names"] for t in row}  # set comprehension
    )
    selected_themes = st.sidebar.multiselect(
        "Themes",
        options=all_themes,
        default=all_themes,
    )

    # Text search
    search_query = st.sidebar.text_input("Search in review text", value="")

    # Apply filters
    filtered_df = df.copy()

    if selected_sentiments:
        filtered_df = filtered_df[filtered_df["sentiment"].isin(selected_sentiments)]

    if selected_themes and len(selected_themes) != len(all_themes):
        filtered_df = filtered_df[
            filtered_df["theme_names"].apply(
                lambda ts: any(t in selected_themes for t in ts)
            )
        ]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["text"].str.lower().str.contains(q, na=False)
        ]

    # Top-level stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        st.metric("Filtered Reviews", len(filtered_df))
    with col3:
        sentiment_counts = filtered_df["sentiment"].value_counts()
        top_sent = sentiment_counts.index[0] if not sentiment_counts.empty else "N/A"
        st.metric("Most Common Sentiment (filtered)", top_sent)

    st.markdown("---")

    # List of reviews with expandable details
    st.subheader("Reviews & Recommendations")

    if filtered_df.empty:
        st.info("No reviews match the current filters.")
        return

    # Allow user to control how many to show
    max_rows = st.slider("Max reviews to display", min_value=1, max_value=50, value=10)

    for idx, row in filtered_df.head(max_rows).iterrows():
        sentiment = row["sentiment"]
        label = row["label"]
        text = row["text"]
        themes = row["themes"]
        actions = row["actions"]

        # Short preview for header
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "..."

        with st.expander(f"[{sentiment.upper()}] {preview}"):
            st.write("**Original Label:**", label)
            st.write("**Full Review:**")
            st.write(text)

            # Themes
            st.write("**Detected Themes:**")
            if themes:
                for t in themes:
                    name = t.get("name", "unknown")
                    conf = t.get("confidence", 0.0)
                    st.write(f"- `{name}` (confidence: {conf:.2f})")
            else:
                st.write("_No themes detected_")

            # Actions
            st.write("**Recommended Actions:**")
            if actions:
                for a in actions:
                    st.write(f"- **{a.get('title')}** "
                             f"(ID: `{a.get('action_id')}`, "
                             f"Team: `{a.get('team')}`, "
                             f"Priority: {a.get('priority')})")
                    st.write(f"  - Reason: {a.get('reason')}")
            else:
                st.write("_No actions recommended_")

    st.markdown("---")
    st.caption("Data source: local CSV → processed with rule-based pipeline.")


if __name__ == "__main__":
    main()
