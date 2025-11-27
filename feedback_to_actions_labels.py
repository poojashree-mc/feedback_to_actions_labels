import os
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd


# =========  CONFIG  =========

# Path to your sample CSV (change if needed)
DATA_PATH = os.path.join("data", "amazon_sample.csv")

# Number of rows to process (set to None to process everything)
MAX_ROWS = 1000


# =========  DATA MODELS  =========

@dataclass
class Theme:
    name: str
    confidence: float


@dataclass
class ActionRecommendation:
    action_id: str
    title: str
    team: str
    priority: int    # 1 = highest
    reason: str


# =========  THEME KEYWORDS & ACTION CATALOG  =========

# Simple keyword-based themes
THEME_KEYWORDS: Dict[str, List[str]] = {
    "delivery": [
        "delivery", "delivered", "shipping", "ship", "courier", "arrive", "arrived",
        "late", "delay", "delayed", "package", "parcel"
    ],
    "pricing": [
        "price", "cost", "expensive", "cheap", "overpriced", "value", "worth",
        "deal", "discount", "offer"
    ],
    "product_quality": [
        "quality", "defect", "broken", "damaged", "faulty", "durable", "flimsy",
        "material", "wear", "tear"
    ],
    "customer_support": [
        "support", "customer service", "service", "helpdesk", "agent", "rude",
        "polite", "call center", "email support", "chat support"
    ],
    "app_performance": [
        "app", "crash", "bug", "lag", "slow", "performance", "freeze", "error",
        "update", "install"
    ],
    "ux_usability": [
        "easy to use", "difficult to use", "user friendly", "interface", "navigation",
        "menu", "layout", "confusing"
    ],
    "refunds_billing": [
        "refund", "refunded", "return", "returned", "money back", "billing",
        "charged", "charge", "payment", "invoice"
    ],
    "features": [
        "feature", "function", "option", "settings", "missing feature", "lack of",
        "doesn't have", "does not have"
    ],
    "security_privacy": [
        "security", "secure", "privacy", "data", "leak", "hacked", "breach"
    ],
}

# Action catalog mapped to themes
ACTIONS_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "IMPROVE_DELIVERY_SLA",
        "title": "Improve Delivery Partner SLAs",
        "theme": "delivery",
        "team": "Operations",
        "description": "Work with delivery partners to improve speed and reliability.",
    },
    {
        "id": "IMPROVE_DELIVERY_COMMUNICATION",
        "title": "Improve Delivery Time Communication",
        "theme": "delivery",
        "team": "Operations",
        "description": "Update order tracking and ETA messaging to set correct expectations.",
    },
    {
        "id": "REVISE_PRICING_COMMUNICATION",
        "title": "Revise Pricing & Discount Communication",
        "theme": "pricing",
        "team": "Marketing",
        "description": "Clarify pricing structure and highlight value to reduce confusion.",
    },
    {
        "id": "REVIEW_PRODUCT_QUALITY",
        "title": "Review Product Quality & Vendor QA",
        "theme": "product_quality",
        "team": "Merchandising",
        "description": "Investigate product quality issues and enforce quality checks.",
    },
    {
        "id": "TRAIN_SUPPORT_TEAM",
        "title": "Train Customer Support Team",
        "theme": "customer_support",
        "team": "Customer Support",
        "description": "Improve scripts, empathy, and resolution processes.",
    },
    {
        "id": "IMPROVE_REFUND_PROCESS",
        "title": "Simplify Refund & Return Process",
        "theme": "refunds_billing",
        "team": "Finance/Support",
        "description": "Make refunds faster, clearer, and easier for customers.",
    },
    {
        "id": "IMPROVE_APP_STABILITY",
        "title": "Improve App Performance & Stability",
        "theme": "app_performance",
        "team": "Engineering",
        "description": "Fix crashes, lags, and bugs reported by customers.",
    },
    {
        "id": "ENHANCE_USABILITY",
        "title": "Enhance UX & Usability",
        "theme": "ux_usability",
        "team": "Product/Design",
        "description": "Simplify flows and improve user-friendly navigation.",
    },
    {
        "id": "ENHANCE_FEATURE_EDUCATION",
        "title": "Launch Customer Education on Features",
        "theme": "features",
        "team": "Product/Marketing",
        "description": "Educate users on key features via guides and campaigns.",
    },
]


# =========  CORE LOGIC  =========

def infer_sentiment_from_label(label: Optional[str]) -> str:
    """
    Map your dataset labels to sentiment.
    Example from your sample:
      __label__1 -> negative
      __label__2 -> positive
    """
    if label is None:
        return "unknown"
    label = str(label).strip()
    if label == "__label__1":
        return "negative"
    if label == "__label__2":
        return "positive"
    # fallback if more labels exist
    return "unknown"


def detect_themes(text: str) -> List[Theme]:
    """
    Simple keyword-based theme detection.
    Returns a list of Theme(name, confidence).
    """
    text_lower = text.lower()
    themes_found: List[Theme] = []

    for theme_name, keywords in THEME_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            if kw in text_lower:
                hits += 1
        if hits > 0:
            confidence = min(1.0, hits / 3.0)  # crude heuristic
            themes_found.append(Theme(name=theme_name, confidence=confidence))

    # Sort by confidence (descending)
    themes_found.sort(key=lambda t: t.confidence, reverse=True)

    # Keep top 3
    return themes_found[:3]


def map_theme_to_actions(theme_name: str) -> List[Dict[str, Any]]:
    return [a for a in ACTIONS_CATALOG if a["theme"] == theme_name]


def recommend_actions_for_review(
    sentiment: str,
    themes: List[Theme]
) -> List[ActionRecommendation]:
    """
    Rule-based action recommendation:
    - For each theme, pick up to 1–2 actions from the catalog.
    - Negative sentiment -> higher priority
    - Positive -> lower priority (e.g., improvements, not urgent)
    """
    recommendations: List[ActionRecommendation] = []

    if not themes:
        return recommendations

    for idx, theme in enumerate(themes):
        theme_actions = map_theme_to_actions(theme.name)
        if not theme_actions:
            continue

        for k, action in enumerate(theme_actions[:2]):
            if sentiment == "negative":
                priority = 1 + idx + k
            elif sentiment == "neutral":
                priority = 2 + idx + k
            else:  # positive or unknown
                priority = 3 + idx + k

            reason = (
                f"Theme '{theme.name}' detected with confidence {theme.confidence:.2f} "
                f"and sentiment '{sentiment}'."
            )

            recommendations.append(
                ActionRecommendation(
                    action_id=action["id"],
                    title=action["title"],
                    team=action["team"],
                    priority=priority,
                    reason=reason,
                )
            )

    # Sort by priority (1 = highest)
    recommendations.sort(key=lambda r: r.priority)
    return recommendations[:3]


# =========  PIPELINE: LOAD → ANALYZE → SAVE  =========

def load_reviews_with_labels(csv_path: str, max_rows: Optional[int] = MAX_ROWS) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at: {csv_path}\n"
            f"Make sure your file (with columns 'label,text') is there."
        )

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Expecting "label" and "text"
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected 'label' and 'text' columns. Found: {list(df.columns)}"
        )

    # Drop row very text empty or not avaiable.
    df = df[["label", "text"]]
    df = df.dropna(subset=["text"])

    if max_rows is not None:
        df = df.head(max_rows)

    print(f"Loaded {len(df)} rows.")
    return df


def process_feedback(df: pd.DataFrame) -> pd.DataFrame:
    processed_rows = []

    for idx, row in df.iterrows():
        text = str(row["text"])
        label = row["label"]

        sentiment = infer_sentiment_from_label(label)
        themes = detect_themes(text)
        actions = recommend_actions_for_review(sentiment, themes)

        row_out = {
            "label": label,
            "text": text,
            "sentiment": sentiment,
            "themes_json": json.dumps([asdict(t) for t in themes], ensure_ascii=False),
            "actions_json": json.dumps([asdict(a) for a in actions], ensure_ascii=False),
        }
        processed_rows.append(row_out)

    return pd.DataFrame(processed_rows)


def main():
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)

    # 1. Load reviews (label,text)
    df_reviews = load_reviews_with_labels(DATA_PATH, MAX_ROWS)

    # 2. Process (sentiment, themes, actions)
    df_processed = process_feedback(df_reviews)

    # 3. Save to CSV
    output_path = "output_feedback_actions.csv"
    df_processed.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved processed feedback with actions to: {output_path}")

    # 4. Show a few sample rows
    print("\nSample rows:")
    print(df_processed.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
