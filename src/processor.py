"""
processor.py
Core pipeline: load CSV → detect sentiment & themes → recommend actions → save output.
"""

import os
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd


# =========  CONFIG  =========

DATA_PATH = os.path.join("data", "input", "amazon_sample.csv")
OUTPUT_PATH = os.path.join("data", "output", "output_feedback_actions.csv")
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
    priority: int   # 1 = highest
    reason: str


# =========  THEME KEYWORDS & ACTION CATALOG  =========

THEME_KEYWORDS: Dict[str, List[str]] = {
    "delivery": [
        "delivery", "delivered", "shipping", "ship", "courier", "arrive", "arrived",
        "late", "delay", "delayed", "package", "parcel",
    ],
    "pricing": [
        "price", "cost", "expensive", "cheap", "overpriced", "value", "worth",
        "deal", "discount", "offer",
    ],
    "product_quality": [
        "quality", "defect", "broken", "damaged", "faulty", "durable", "flimsy",
        "material", "wear", "tear",
    ],
    "customer_support": [
        "support", "customer service", "service", "helpdesk", "agent", "rude",
        "polite", "call center", "email support", "chat support",
    ],
    "app_performance": [
        "app", "crash", "bug", "lag", "slow", "performance", "freeze", "error",
        "update", "install",
    ],
    "ux_usability": [
        "easy to use", "difficult to use", "user friendly", "interface", "navigation",
        "menu", "layout", "confusing",
    ],
    "refunds_billing": [
        "refund", "refunded", "return", "returned", "money back", "billing",
        "charged", "charge", "payment", "invoice",
    ],
    "features": [
        "feature", "function", "option", "settings", "missing feature", "lack of",
        "doesn't have", "does not have",
    ],
    "security_privacy": [
        "security", "secure", "privacy", "data", "leak", "hacked", "breach",
    ],
}

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
    """Map dataset labels to sentiment strings."""
    if label is None:
        return "unknown"
    label = str(label).strip()
    mapping = {"__label__1": "negative", "__label__2": "positive"}
    return mapping.get(label, "unknown")


def detect_themes(text: str) -> List[Theme]:
    """Keyword-based theme detection. Returns top 3 themes by confidence."""
    text_lower = text.lower()
    found: List[Theme] = []

    for theme_name, keywords in THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            confidence = min(1.0, hits / 3.0)
            found.append(Theme(name=theme_name, confidence=confidence))

    found.sort(key=lambda t: t.confidence, reverse=True)
    return found[:3]


def _actions_for_theme(theme_name: str) -> List[Dict[str, Any]]:
    return [a for a in ACTIONS_CATALOG if a["theme"] == theme_name]


def recommend_actions(sentiment: str, themes: List[Theme]) -> List[ActionRecommendation]:
    """
    Rule-based action recommendations.
    Negative sentiment → higher priority (lower number).
    """
    recommendations: List[ActionRecommendation] = []
    priority_base = {"negative": 1, "neutral": 2}.get(sentiment, 3)

    for idx, theme in enumerate(themes):
        for k, action in enumerate(_actions_for_theme(theme.name)[:2]):
            recommendations.append(
                ActionRecommendation(
                    action_id=action["id"],
                    title=action["title"],
                    team=action["team"],
                    priority=priority_base + idx + k,
                    reason=(
                        f"Theme '{theme.name}' detected with confidence "
                        f"{theme.confidence:.2f} and sentiment '{sentiment}'."
                    ),
                )
            )

    recommendations.sort(key=lambda r: r.priority)
    return recommendations[:3]


# =========  PIPELINE  =========

def load_reviews(csv_path: str, max_rows: Optional[int] = MAX_ROWS) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[["label", "text"]].dropna(subset=["text"])
    if max_rows:
        df = df.head(max_rows)

    print(f"Loaded {len(df)} rows.")
    return df


def process_feedback(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = str(row["text"])
        label = row["label"]
        sentiment = infer_sentiment_from_label(label)
        themes = detect_themes(text)
        actions = recommend_actions(sentiment, themes)

        rows.append({
            "label": label,
            "text": text,
            "sentiment": sentiment,
            "themes_json": json.dumps([asdict(t) for t in themes], ensure_ascii=False),
            "actions_json": json.dumps([asdict(a) for a in actions], ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df_reviews = load_reviews(DATA_PATH, MAX_ROWS)
    df_processed = process_feedback(df_reviews)
    df_processed.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSaved output to: {OUTPUT_PATH}")
    print("\nSample rows:")
    print(df_processed.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
