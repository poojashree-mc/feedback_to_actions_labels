# Feedback to Actions Labels

A rule-based pipeline that processes customer feedback, classifies sentiment and themes, and recommends actionable next steps — with a Streamlit dashboard for exploring results.

---

## How It Works

1. **Load** — reads a CSV of customer reviews (columns: `label`, `text`)
2. **Sentiment** — maps labels (`__label__1` / `__label__2`) to `negative` / `positive`
3. **Theme detection** — keyword matching across 9 themes (delivery, pricing, app performance, etc.)
4. **Action recommendations** — rule-based catalog assigns prioritised actions per theme and sentiment
5. **Output** — saves an enriched CSV; visualised in a Streamlit dashboard

---

## Project Structure

```
feedback_to_actions_labels/
├── src/
│   ├── __init__.py
│   ├── processor.py          ← core pipeline (load → analyse → save)
│   └── dashboard.py          ← Streamlit dashboard
├── data/
│   ├── input/                ← place your source CSV here
│   └── output/               ← generated output lands here
├── tests/
│   └── test_processor.py     ← unit tests for core logic
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your input data

Place a CSV file at `data/input/amazon_sample.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `label` | Sentiment label (`__label__1` = negative, `__label__2` = positive) |
| `text`  | Raw customer review text |

### 3. Run the pipeline

```bash
python src/processor.py
```

Output is saved to `data/output/output_feedback_actions.csv`.

### 4. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

---

## Output Format

The output CSV contains one row per review with these columns:

| Column | Description |
|--------|-------------|
| `label` | Original label from input |
| `text` | Review text |
| `sentiment` | Inferred sentiment (`positive` / `negative` / `unknown`) |
| `themes_json` | JSON array of detected themes with confidence scores |
| `actions_json` | JSON array of recommended actions with team and priority |

---

## Themes Detected

| Theme | Examples keywords |
|-------|------------------|
| `delivery` | shipping, delayed, courier, parcel |
| `pricing` | expensive, discount, value, overpriced |
| `product_quality` | defect, broken, damaged, faulty |
| `customer_support` | helpdesk, rude, agent, call center |
| `app_performance` | crash, bug, lag, freeze |
| `ux_usability` | confusing, navigation, layout, interface |
| `refunds_billing` | refund, return, charged, invoice |
| `features` | missing feature, option, settings |
| `security_privacy` | data, hacked, breach, privacy |

---

## Running Tests

```bash
python tests/test_processor.py
```

---

## Configuration

Edit the constants at the top of `src/processor.py` to adjust behaviour:

| Constant | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/input/amazon_sample.csv` | Path to input CSV |
| `OUTPUT_PATH` | `data/output/output_feedback_actions.csv` | Path to output CSV |
| `MAX_ROWS` | `1000` | Max rows to process (`None` = all) |
