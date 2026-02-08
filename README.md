# Streamlit Analytics Dashboard

## Overview
Interactive analytics dashboard for uploaded CSV files with business framing, data cleaning, filtering, visualizations, findings, comparison, and exports.

## Setup
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

## Run
`streamlit run app.py --server.port 8501 --server.address 127.0.0.1`

## Expected input format
- CSV with headers
- Mixed numeric/categorical/datetime columns supported
- Datetime columns are auto-coerced when parse confidence is high

## Core features
- Business context and KPI framing
- Data source/provenance and schema summary
- Sidebar filters for categorical, numeric, and datetime columns
- Data cleaning actions with cleaning history
- Plotly visualizations (histogram, bar, correlation, time series, scatter)
- Comparison and benchmark tab for filtered vs full dataset
- Findings with multi-insight narrative output
- Export options: CSV, JSON summary, HTML summary, README template

## Troubleshooting
- Upload errors: confirm CSV format and file size under 200MB
- Empty results: clear filters or loosen filter ranges
- Missing charts: ensure required column types are present
