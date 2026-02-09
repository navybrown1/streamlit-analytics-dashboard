# Business Analytics Dashboard

## Overview
AI-powered interactive analytics dashboard for uploaded CSV files with business framing, data cleaning, filtering, visualizations, findings, comparison, anomaly detection, trend forecasting, and exports.

## Live Demo
**[Launch the app on Streamlit Cloud](https://app-analytics-dashboard-bk9byiwxmuvgemwedifjn7.streamlit.app/)**

## Sample Dataset
Use any CSV file, or try one of these free public datasets:
- [Tesla Trip Data (Kaggle)](https://www.kaggle.com/datasets) — search "Tesla trip data" or "EV trip logs"
- [NYC Taxi Trip Data (sample)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Superstore Sales (Kaggle)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — classic business analytics dataset

Any CSV with headers will work. Best results with mixed numeric, categorical, and datetime columns.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

## Expected Input Format
- CSV file with headers (up to 200MB)
- Mixed numeric, categorical, and datetime columns supported
- Datetime columns are auto-detected when parse confidence exceeds 85%
- Encoding fallbacks: UTF-8, UTF-8-SIG, Latin-1
- Delimiter fallbacks: auto-detect, comma, semicolon, tab, pipe

## Features

### Core Analytics
- Business question templates with decision-support framing
- 5 auto-computed KPIs (trip count, duration, efficiency, battery drop, top route)
- Data quality diagnostics: missing values, duplicates, column types, descriptive stats
- Missing data heatmap visualization
- Sidebar filters for categorical (multi-select + text search), numeric (range slider), and datetime columns
- Plotly visualizations: histogram, bar chart, correlation heatmap, time series, scatter explorer
- Filtered vs full dataset comparison with time period overlay and benchmark tracking
- Auto-generated findings section with 7 data-grounded insights
- Export: filtered CSV, JSON summary, HTML report, README template

### AI-Powered (requires free Google Gemini API key)
- Natural language data chat — ask questions about your data in plain English
- One-click AI narrative report generation (executive summary, findings, recommendations)
- Only column names, summary stats, and 5 sample rows are sent to the API — never the full dataset

### Built-in ML (no API key needed)
- Anomaly detection via Isolation Forest (adjustable sensitivity)
- Trend forecasting with polynomial regression and 95% confidence bands

## AI Setup (Optional)
To enable AI chat and reports, set your free Google Gemini API key:

**Local:** Set environment variable `GOOGLE_API_KEY=your-key-here`

**Streamlit Cloud:** Add to Settings > Secrets:
```toml
GOOGLE_API_KEY = "your-key-here"
```

Get a free key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Troubleshooting
- **Upload errors:** Confirm file is CSV format and under 200MB
- **Empty charts:** Verify required numeric/categorical columns exist after cleaning
- **Empty results:** Use "Reset all filters" or loosen filter criteria
- **AI errors:** Check that your Gemini API key is valid and has free-tier quota
- **Unexpected results:** Review cleaning history and active filters in the Data Stats expander
