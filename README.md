
# Feedback to Actions Labels – Project

This project processes customer feedback data and converts it into structured actionable labels. It includes a Streamlit dashboard for visualizing insights and exporting results.

## Features
- Processes raw feedback.
- Cleans and labels text using ML logic.
- Streamlit dashboard for analysis.
- Generates `output_feedback_actions.csv`.

## Project Structure
```
feedback_to_actions_labels-main/
│── data/
│── feedback_to_actions_labels.py
│── feedback_dashboard_streamlit.py
│── output_feedback_actions.csv
│── README.md
```

## Installation
```
pip install pandas numpy streamlit scikit-learn nltk
```

## Run Script
```
python feedback_to_actions_labels.py
```

## Run Dashboard
```
streamlit run feedback_dashboard_streamlit.py
```

## Input Format
CSV with columns:
- feedback_id
- customer_feedback

## Output
Labeled CSV with:
- feedback
- action_label
- priority


