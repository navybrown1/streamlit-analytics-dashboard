from __future__ import annotations

import io
import json
import re
import warnings
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


MAX_UPLOAD_MB = 200

st.set_page_config(page_title="Local Business Analytics Dashboard", page_icon="📊", layout="wide")
st.title("Local Business Analytics Dashboard")
st.caption("Upload a CSV to explore business context, data quality, filters, insights, and exports.")


def safe_widget_key(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value)


def clear_filter_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("flt_"):
            del st.session_state[key]


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Read CSV from bytes with encoding and delimiter fallbacks."""
    meta: dict[str, str] = {}
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    delimiters: list[str | None] = [None, ",", ";", "\t", "|"]

    for enc in encodings:
        for sep in delimiters:
            try:
                buf = io.BytesIO(file_bytes)
                if sep is None:
                    df = pd.read_csv(buf, sep=None, engine="python", encoding=enc)
                    delim_label = "auto"
                else:
                    df = pd.read_csv(buf, sep=sep, encoding=enc)
                    delim_label = "tab" if sep == "\t" else sep
                meta["encoding"] = enc
                meta["delimiter"] = delim_label
                return df, meta
            except Exception:
                continue

    raise ValueError("Could not parse the CSV with supported encodings and delimiters.")


def make_unique_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    counts: dict[str, int] = {}
    new_cols: list[str] = []
    renamed: list[tuple[str, str]] = []

    for col in df.columns:
        seen = counts.get(col, 0)
        counts[col] = seen + 1
        if seen == 0:
            new_col = col
        else:
            new_col = f"{col}__{seen + 1}"
            renamed.append((col, new_col))
        new_cols.append(new_col)

    out = df.copy()
    out.columns = new_cols
    return out, renamed


def coerce_datetime_columns(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Convert object columns that parse as datetimes above a threshold."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        sample = out[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if float(parsed.notna().mean()) >= threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def column_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    other_cols = [
        col
        for col in df.columns
        if col not in set(numeric_cols + datetime_cols + categorical_cols)
    ]
    return numeric_cols, categorical_cols, datetime_cols, other_cols


def dataset_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    unique_values = df.nunique(dropna=True)
    dtype = df.dtypes.astype(str)

    def recommendation(dtype_str: str, miss_pct: float) -> str:
        if miss_pct == 0:
            return "No action needed"
        dtype_low = dtype_str.lower()
        if "float" in dtype_low or "int" in dtype_low:
            return "Median fill (high missing)" if miss_pct >= 20 else "Mean/median fill"
        if "datetime" in dtype_low:
            return "Forward fill or drop if sparse"
        return "Mode or custom label"

    rec = [recommendation(str(dtype[col]), float(missing_pct[col])) for col in df.columns]

    overview = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": dtype.values,
            "non_null": (df.shape[0] - missing).values,
            "missing_count": missing.values,
            "missing_pct": missing_pct.values,
            "unique_values": unique_values.values,
            "imputation_recommendation": rec,
        }
    )
    return overview.sort_values(["missing_pct", "unique_values"], ascending=[False, False])


def numeric_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["column", "mean", "median", "std_dev", "min", "max"])
    data = df[numeric_cols]
    stats = pd.DataFrame(
        {
            "mean": data.mean(numeric_only=True),
            "median": data.median(numeric_only=True),
            "std_dev": data.std(numeric_only=True),
            "min": data.min(numeric_only=True),
            "max": data.max(numeric_only=True),
        }
    )
    return stats.reset_index().rename(columns={"index": "column"})


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        hit = lowered.get(candidate.lower())
        if hit:
            return hit
    for candidate in candidates:
        candidate_l = candidate.lower()
        for col in columns:
            if candidate_l in col.lower():
                return col
    return None


def infer_schema_description(column_name: str) -> str:
    col = column_name.lower()
    if "start" in col and "location" in col:
        return "Trip starting location"
    if "end" in col and "location" in col:
        return "Trip ending location"
    if "duration" in col:
        return "Trip duration measure"
    if "battery" in col:
        return "Battery state metric"
    if "energy" in col or "wh/mi" in col or "kwh" in col:
        return "Energy consumption metric"
    if "distance" in col or "odometer" in col:
        return "Distance or mileage metric"
    if "speed" in col:
        return "Speed metric"
    if "temp" in col:
        return "Temperature metric"
    if "date" in col or "time" in col or "started" in col or "ended" in col:
        return "Timestamp/date field"
    if "tag" in col:
        return "User trip label"
    return "General data field"


def guess_export_date(text: str) -> str:
    month_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\\s+\\d{1,2}\\s+\\d{4}",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        try:
            return datetime.strptime(month_match.group(0), "%B %d %Y").date().isoformat()
        except Exception:
            return datetime.now().date().isoformat()

    iso_match = re.search(r"\\d{4}-\\d{2}-\\d{2}", text)
    if iso_match:
        return iso_match.group(0)

    return datetime.now().date().isoformat()


def calculate_business_kpis(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    kpis: list[tuple[str, str, str]] = []
    cols = df.columns.tolist()

    duration_col = find_column(cols, ["Duration (Minutes)", "duration"])
    avg_energy_col = find_column(cols, ["Average Energy Used (Wh/mi)", "wh/mi", "energy used"])
    start_col = find_column(cols, ["Starting Location", "start location"])
    end_col = find_column(cols, ["Ending Location", "end location"])
    start_battery_col = find_column(cols, ["Starting Battery (%)", "start battery"])
    end_battery_col = find_column(cols, ["Ending Battery (%)", "end battery"])
    distance_col = find_column(cols, ["Distance (mi)", "distance"])

    kpis.append(("Total Trips", f"{len(df):,}", "Total records in current cleaned dataset"))

    if duration_col and pd.api.types.is_numeric_dtype(df[duration_col]):
        kpis.append(
            (
                "Avg Trip Duration",
                f"{df[duration_col].dropna().mean():.1f} min",
                "Average trip time across filtered dataset",
            )
        )

    if avg_energy_col and pd.api.types.is_numeric_dtype(df[avg_energy_col]):
        kpis.append(
            (
                "Avg Trip Efficiency",
                f"{df[avg_energy_col].dropna().mean():.1f} Wh/mi",
                "Lower Wh/mi usually indicates better efficiency",
            )
        )

    if (
        start_battery_col
        and end_battery_col
        and distance_col
        and pd.api.types.is_numeric_dtype(df[start_battery_col])
        and pd.api.types.is_numeric_dtype(df[end_battery_col])
        and pd.api.types.is_numeric_dtype(df[distance_col])
    ):
        subset = df[[start_battery_col, end_battery_col, distance_col]].dropna().copy()
        subset = subset[subset[distance_col] > 0]
        if not subset.empty:
            rate = ((subset[start_battery_col] - subset[end_battery_col]) / subset[distance_col]) * 100
            if not rate.dropna().empty:
                kpis.append(
                    (
                        "Battery Drop / 100 mi",
                        f"{rate.dropna().mean():.2f}%",
                        "Estimated battery percentage consumed per 100 miles",
                    )
                )

    if start_col and end_col:
        route_series = (
            df[start_col].fillna("(Missing)").astype(str)
            + " → "
            + df[end_col].fillna("(Missing)").astype(str)
        )
        if not route_series.empty:
            top_route = route_series.value_counts().head(1)
            if not top_route.empty:
                kpis.append(
                    (
                        "Most Frequent Route",
                        top_route.index[0][:24],
                        f"{int(top_route.iloc[0]):,} trip(s) for top route",
                    )
                )

    return kpis[:5]


def apply_fill_missing(
    df: pd.DataFrame,
    numeric_strategy: str,
    categorical_strategy: str,
    numeric_custom: str,
    categorical_custom: str,
) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    filled_cells = 0

    if numeric_strategy != "None":
        for col in out.select_dtypes(include=["number"]).columns:
            missing_count = int(out[col].isna().sum())
            if missing_count == 0:
                continue

            if numeric_strategy == "Mean":
                fill_value = out[col].mean()
            elif numeric_strategy == "Median":
                fill_value = out[col].median()
            elif numeric_strategy == "Mode":
                modes = out[col].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else pd.NA
            else:
                fill_value = pd.to_numeric(pd.Series([numeric_custom]), errors="coerce").iloc[0]

            if pd.isna(fill_value):
                continue
            out[col] = out[col].fillna(fill_value)
            filled_cells += missing_count

    if categorical_strategy != "None":
        cat_cols = out.select_dtypes(include=["object", "category", "bool", "string"]).columns
        for col in cat_cols:
            missing_count = int(out[col].isna().sum())
            if missing_count == 0:
                continue

            if categorical_strategy == "Mode":
                modes = out[col].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            else:
                fill_value = categorical_custom if categorical_custom.strip() else "Unknown"

            out[col] = out[col].fillna(fill_value)
            filled_cells += missing_count

    return out, filled_cells


def clean_text_columns(
    df: pd.DataFrame,
    text_cols: list[str],
    trim_whitespace: bool,
    remove_special_chars: bool,
) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    changed_cells = 0

    for col in text_cols:
        if col not in out.columns:
            continue

        mask = out[col].notna()
        series = out.loc[mask, col].astype(str)
        before = series.copy()

        if trim_whitespace:
            series = series.str.strip()
        if remove_special_chars:
            series = series.str.replace(r"[^0-9a-zA-Z\\s\\-\\.,:/()&]", "", regex=True)

        changed_cells += int((before != series).sum())
        out.loc[mask, col] = series

    return out, changed_cells


def filtered_date_window(df: pd.DataFrame, datetime_cols: list[str]) -> str:
    for col in datetime_cols:
        series = df[col].dropna()
        if not series.empty:
            return f"{series.min().date()} to {series.max().date()}"
    return "No datetime range"


def sample_size_label(full_df: pd.DataFrame, filtered_df: pd.DataFrame, datetime_cols: list[str]) -> str:
    total = len(full_df)
    current = len(filtered_df)
    pct = (current / total * 100) if total else 0
    date_window = filtered_date_window(filtered_df, datetime_cols)
    return f"Sample size: n = {current:,} ({pct:.1f}% of full dataset). Date range: {date_window}."


def get_palette_sequence(name: str) -> list[str]:
    palettes: dict[str, list[str]] = {
        "Plotly": px.colors.qualitative.Plotly,
        "Safe": px.colors.qualitative.Safe,
        "Bold": px.colors.qualitative.Bold,
        "Dark24": px.colors.qualitative.Dark24,
        "Set2": px.colors.qualitative.Set2,
    }
    return palettes.get(name, px.colors.qualitative.Plotly)


def comparison_metrics_table(
    df_full: pd.DataFrame, df_filtered: pd.DataFrame, numeric_cols: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in numeric_cols:
        full_series = df_full[col].dropna()
        filtered_series = df_filtered[col].dropna()
        if full_series.empty or filtered_series.empty:
            continue

        full_mean = float(full_series.mean())
        filtered_mean = float(filtered_series.mean())
        delta = filtered_mean - full_mean
        delta_pct = (delta / full_mean * 100) if full_mean != 0 else 0.0

        rows.append(
            {
                "metric": col,
                "full_mean": full_mean,
                "filtered_mean": filtered_mean,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["metric", "full_mean", "filtered_mean", "delta", "delta_pct"])

    out = pd.DataFrame(rows)
    return out.sort_values("delta_pct", key=lambda s: s.abs(), ascending=False)


def build_readme_text() -> str:
    return """# Streamlit Analytics Dashboard

## Overview
Interactive analytics dashboard for uploaded CSV files with business framing, filtering, cleaning, visualizations, findings, and exports.

## Setup
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

## Run
`streamlit run app.py --server.port 8501 --server.address 127.0.0.1`

## Input Format
- Any CSV schema is supported.
- Best results when headers are present.
- Datetime columns are auto-detected where possible.

## Key Features
- Data quality diagnostics and missing-value recommendations
- Sidebar filters for categorical, numeric, and datetime columns
- Plotly histogram, bar chart, correlation, and time series views
- Insight generation and summary export (CSV, JSON, HTML)
- Cleaning actions: drop missing, deduplicate, fill missing, type conversion, text cleaning

## Troubleshooting
- If upload fails, confirm file is CSV and below 200MB.
- If filters return no rows, click \"Clear all filters\".
- If charts are empty, verify at least one numeric/categorical column exists.
"""


def apply_filters_sidebar(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    st.sidebar.header("Filters")
    st.sidebar.caption("Choose columns to filter. Each filter narrows the current cleaned dataset.")

    if st.sidebar.button("Clear all filters", use_container_width=True):
        clear_filter_state()
        st.rerun()

    filterable_cols = categorical_cols + numeric_cols + datetime_cols
    selected_cols = st.sidebar.multiselect(
        "Filter columns",
        options=filterable_cols,
        default=[],
        key="flt_selected_columns",
    )

    filtered = df.copy()
    summaries: list[str] = []

    for col in selected_cols:
        key_base = safe_widget_key(col)

        if col in categorical_cols:
            st.sidebar.markdown(f"**{col}**")
            series = filtered[col].fillna("(Missing)").astype(str)
            unique_count = int(series.nunique(dropna=False))
            options = series.value_counts().head(75).index.tolist()

            st.sidebar.caption(f"Unique values: {unique_count:,}")

            chosen = st.sidebar.multiselect(
                f"Values for {col}",
                options=options,
                default=[],
                key=f"flt_cat_vals_{key_base}",
            )
            contains = st.sidebar.text_input(
                f"Contains text in {col}",
                value="",
                key=f"flt_cat_contains_{key_base}",
            )

            if chosen:
                filtered = filtered[filtered[col].fillna("(Missing)").astype(str).isin(chosen)]
                summaries.append(f"{col}: {len(chosen)} value(s)")
            if contains.strip():
                needle = contains.strip()
                filtered = filtered[
                    filtered[col].fillna("").astype(str).str.contains(needle, case=False, na=False)
                ]
                summaries.append(f"{col} contains '{needle}'")

            st.sidebar.caption(f"Rows after {col}: {len(filtered):,}")

        elif col in numeric_cols:
            st.sidebar.markdown(f"**{col}**")
            series = filtered[col].dropna()
            if series.empty:
                st.sidebar.info(f"{col}: no numeric values available.")
                continue

            min_val = float(series.min())
            max_val = float(series.max())
            if min_val == max_val:
                st.sidebar.caption(f"{col}: constant value {min_val:.3g}")
                continue

            default_range = (min_val, max_val)
            rng = st.sidebar.slider(
                f"Range for {col}",
                min_value=min_val,
                max_value=max_val,
                value=default_range,
                key=f"flt_num_rng_{key_base}",
            )
            keep_missing = st.sidebar.checkbox(
                f"Keep missing values for {col}",
                value=True,
                key=f"flt_num_keepna_{key_base}",
            )

            mask = filtered[col].between(rng[0], rng[1], inclusive="both")
            if keep_missing:
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

            if rng != default_range or not keep_missing:
                summaries.append(f"{col}: {rng[0]:.3g} to {rng[1]:.3g}")

            st.sidebar.caption(f"Rows after {col}: {len(filtered):,}")

        elif col in datetime_cols:
            st.sidebar.markdown(f"**{col}**")
            series = filtered[col].dropna()
            if series.empty:
                st.sidebar.info(f"{col}: no datetime values available.")
                continue

            min_dt = series.min()
            max_dt = series.max()
            if pd.isna(min_dt) or pd.isna(max_dt) or min_dt == max_dt:
                st.sidebar.caption(f"{col}: not enough date variation.")
                continue

            st.sidebar.caption("Format: YYYY-MM-DD to YYYY-MM-DD")
            picked = st.sidebar.date_input(
                f"Date range for {col}",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
                key=f"flt_dt_rng_{key_base}",
            )
            keep_missing = st.sidebar.checkbox(
                f"Keep missing dates for {col}",
                value=True,
                key=f"flt_dt_keepna_{key_base}",
            )

            if isinstance(picked, (list, tuple)) and len(picked) == 2:
                start_date, end_date = picked
                start_ts = pd.to_datetime(start_date)
                end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                mask = (filtered[col] >= start_ts) & (filtered[col] <= end_ts)
                if keep_missing:
                    mask = mask | filtered[col].isna()
                filtered = filtered[mask]

                day_count = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
                summaries.append(f"{col}: {start_date} to {end_date} ({day_count} days)")

            st.sidebar.caption(f"Rows after {col}: {len(filtered):,}")

    return filtered, summaries


def generate_findings(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> list[str]:
    insights: list[str] = []

    total_rows = len(df_full)
    filtered_rows = len(df_filtered)
    if total_rows:
        pct = filtered_rows / total_rows * 100
        insights.append(f"Viewing **{filtered_rows:,}** of **{total_rows:,}** rows ({pct:.1f}%).")

    missing_pct = (df_full.isna().mean() * 100).sort_values(ascending=False)
    if not missing_pct.empty and float(missing_pct.iloc[0]) > 0:
        top_missing_col = missing_pct.index[0]
        insights.append(
            f"Most missing field: **{top_missing_col}** at **{float(missing_pct.iloc[0]):.1f}%** missing."
        )

    cols = df_filtered.columns.tolist()
    start_col = find_column(cols, ["Starting Location", "start location"])
    end_col = find_column(cols, ["Ending Location", "end location"])
    duration_col = find_column(cols, ["Duration (Minutes)", "duration"])
    energy_col = find_column(cols, ["Average Energy Used (Wh/mi)", "wh/mi", "energy used"])

    if start_col and not df_filtered.empty:
        counts = df_filtered[start_col].fillna("(Missing)").astype(str).value_counts()
        if not counts.empty:
            share = counts.iloc[0] / counts.sum() * 100
            insights.append(
                f"Most common starting location: **{counts.index[0]}** with **{int(counts.iloc[0]):,} trips ({share:.1f}%)**."
            )

    if end_col and not df_filtered.empty:
        counts = df_filtered[end_col].fillna("(Missing)").astype(str).value_counts()
        if not counts.empty:
            top3_share = counts.head(3).sum() / counts.sum() * 100
            insights.append(f"Top 3 destinations account for **{top3_share:.1f}%** of trips.")

    if duration_col and pd.api.types.is_numeric_dtype(df_filtered[duration_col]):
        durations = df_filtered[duration_col].dropna()
        if not durations.empty:
            avg_dur = float(durations.mean())
            std_dur = float(durations.std()) if durations.shape[0] > 1 else 0.0
            p95_dur = float(durations.quantile(0.95))
            short_pct = float((durations < 10).mean() * 100)
            long_pct = float((durations > 30).mean() * 100)
            insights.append(
                f"Trip duration: average **{avg_dur:.1f} min**, std dev **{std_dur:.1f}**, and 95% of trips are under **{p95_dur:.1f} min**."
            )
            insights.append(
                f"Trip mix: **{short_pct:.1f}%** are short (<10 min) vs **{long_pct:.1f}%** long (>30 min)."
            )

    if energy_col and pd.api.types.is_numeric_dtype(df_filtered[energy_col]):
        energy = df_filtered[energy_col].dropna()
        if not energy.empty:
            median_energy = float(energy.median())
            insights.append(f"Median energy use is **{median_energy:.1f} Wh/mi**.")

    if datetime_cols:
        dt_col = datetime_cols[0]
        dt_series = df_filtered[dt_col].dropna()
        if not dt_series.empty:
            weekend_pct = float((dt_series.dt.dayofweek >= 5).mean() * 100)
            insights.append(f"Weekend trips represent **{weekend_pct:.1f}%** of activity in the filtered view.")

            if energy_col and pd.api.types.is_numeric_dtype(df_filtered[energy_col]):
                trend_df = df_filtered[[dt_col, energy_col]].dropna().copy()
                if not trend_df.empty:
                    monthly = trend_df.set_index(dt_col)[energy_col].resample("ME").mean().dropna()
                    if monthly.shape[0] >= 2:
                        trend = "improving" if monthly.iloc[-1] < monthly.iloc[0] else "declining"
                        insights.append(
                            f"Energy efficiency trend appears **{trend}** from **{monthly.index[0].date()}** to **{monthly.index[-1].date()}**."
                        )

    if numeric_cols and not df_filtered.empty:
        outlier_col = numeric_cols[0]
        values = df_filtered[outlier_col].dropna()
        if values.shape[0] >= 8:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((values < lower) | (values > upper)).sum())
                insights.append(
                    f"Detected **{outlier_count:,}** outlier rows in **{outlier_col}** using the IQR rule."
                )

    if len(insights) < 5:
        if categorical_cols and not df_filtered.empty:
            cat = categorical_cols[0]
            uniq = int(df_filtered[cat].fillna("(Missing)").astype(str).nunique())
            insights.append(f"Column **{cat}** has **{uniq:,}** unique categories in the current view.")

    if len(insights) < 5:
        insights.append("Use sidebar filters to isolate route clusters and compare efficiency by segment.")

    return insights[:7]


def build_summary_payload(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    filters: list[str],
    insights: list[str],
    meta: dict,
    source_info: dict,
    datetime_cols: list[str],
) -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "total_rows": int(len(df_full)),
            "filtered_rows": int(len(df_filtered)),
            "columns": int(df_full.shape[1]),
            "encoding": meta.get("encoding", "unknown"),
            "delimiter": meta.get("delimiter", "unknown"),
            "date_window_filtered": filtered_date_window(df_filtered, datetime_cols),
        },
        "source": source_info,
        "filters": filters,
        "insights": insights,
        "visualizations": [
            "Histogram (numeric)",
            "Bar chart (categorical)",
            "Correlation heatmap",
            "Time series",
            "Missing data heatmap",
        ],
    }


def build_html_report(summary: dict) -> str:
    filters_html = "<li>No active filters</li>" if not summary["filters"] else "".join(
        f"<li>{item}</li>" for item in summary["filters"]
    )
    insights_html = "".join(f"<li>{item}</li>" for item in summary["insights"])
    visuals_html = "".join(f"<li>{item}</li>" for item in summary["visualizations"])

    return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Analytics Summary Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ background: #f5f5f5; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Analytics Summary Report</h1>
  <p>Generated: {summary['generated_at']}</p>
  <div class=\"meta\">
    <p><b>Total rows:</b> {summary['dataset']['total_rows']}</p>
    <p><b>Filtered rows:</b> {summary['dataset']['filtered_rows']}</p>
    <p><b>Columns:</b> {summary['dataset']['columns']}</p>
    <p><b>Date window:</b> {summary['dataset']['date_window_filtered']}</p>
    <p><b>Encoding:</b> {summary['dataset']['encoding']} | <b>Delimiter:</b> {summary['dataset']['delimiter']}</p>
  </div>

  <h2>Data Source</h2>
  <ul>
    <li><b>Source:</b> {summary['source'].get('source_system', '')}</li>
    <li><b>Exported on:</b> {summary['source'].get('export_date', '')}</li>
    <li><b>Owner:</b> {summary['source'].get('owner', '')}</li>
    <li><b>Reference:</b> {summary['source'].get('source_url', '')}</li>
    <li><b>Limitations:</b> {summary['source'].get('limitations', '')}</li>
  </ul>

  <h2>Applied Filters</h2>
  <ul>{filters_html}</ul>

  <h2>Key Insights</h2>
  <ul>{insights_html}</ul>

  <h2>Top Visuals</h2>
  <ul>{visuals_html}</ul>
</body>
</html>
""".strip()


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File exceeds {MAX_UPLOAD_MB}MB limit. Please upload a smaller CSV.")
    st.stop()

try:
    raw_bytes = uploaded.getvalue()
    data_raw, meta = read_csv_bytes(raw_bytes)
except Exception as exc:
    st.error("Could not read this file as a CSV. Check delimiter, encoding, and file format.")
    st.exception(exc)
    st.stop()

if data_raw is None or data_raw.shape[1] == 0:
    st.error("This CSV has no columns. Upload a valid dataset.")
    st.stop()

if data_raw.empty:
    st.error("This CSV has no rows. Upload a non-empty dataset.")
    st.stop()

data_raw = data_raw.dropna(how="all").copy()
if data_raw.empty:
    st.error("Your dataset has no usable rows after removing fully empty rows.")
    st.stop()

data_raw, renamed_columns = make_unique_columns(data_raw)
data_raw = coerce_datetime_columns(data_raw)

dataset_token = f"{uploaded.name}:{uploaded.size}"
if st.session_state.get("dataset_token") != dataset_token:
    st.session_state["dataset_token"] = dataset_token
    st.session_state["working_data"] = data_raw.copy()
    st.session_state["cleaning_history"] = []

    default_source = "Tesla app CSV export" if "tesla" in uploaded.name.lower() else "Uploaded CSV file"
    st.session_state["src_system"] = default_source
    st.session_state["src_export_date"] = guess_export_date(uploaded.name)
    st.session_state["src_owner"] = "Edwin Brown"
    st.session_state["src_url"] = "Local file upload"
    st.session_state["src_limitations"] = "Route names and tags may be user-entered and inconsistent."

if renamed_columns:
    st.warning(
        "Duplicate column names were detected and renamed with suffixes (example: __2, __3) to avoid ambiguity."
    )

st.sidebar.header("Data Cleaning")
st.sidebar.caption("Apply optional cleanup actions before filtering and visualization.")

working_data = st.session_state["working_data"]

if st.sidebar.button("Drop rows with missing values", use_container_width=True):
    before = len(working_data)
    working_data = working_data.dropna().copy()
    removed = before - len(working_data)
    st.session_state["cleaning_history"].append(
        f"Dropped rows with missing values: {before:,} -> {len(working_data):,} (removed {removed:,})"
    )

if st.sidebar.button("Remove duplicate rows", use_container_width=True):
    before = len(working_data)
    working_data = working_data.drop_duplicates().copy()
    removed = before - len(working_data)
    st.session_state["cleaning_history"].append(
        f"Removed duplicates: {before:,} -> {len(working_data):,} (removed {removed:,})"
    )

with st.sidebar.expander("Fill missing values", expanded=False):
    numeric_strategy = st.selectbox(
        "Numeric columns",
        options=["None", "Mean", "Median", "Mode", "Custom"],
        key="clean_numeric_strategy",
    )
    numeric_custom = st.text_input("Numeric custom value", value="0", key="clean_numeric_custom")

    categorical_strategy = st.selectbox(
        "Categorical columns",
        options=["None", "Mode", "Custom"],
        key="clean_categorical_strategy",
    )
    categorical_custom = st.text_input(
        "Categorical custom value", value="Unknown", key="clean_categorical_custom"
    )

    if st.button("Apply fill-missing", key="clean_apply_fill", use_container_width=True):
        updated, filled_cells = apply_fill_missing(
            working_data,
            numeric_strategy,
            categorical_strategy,
            numeric_custom,
            categorical_custom,
        )
        working_data = updated
        st.session_state["cleaning_history"].append(
            f"Filled missing values: {filled_cells:,} cell(s) updated"
        )

with st.sidebar.expander("Data type conversion", expanded=False):
    convert_col = st.selectbox(
        "Column to convert",
        options=working_data.columns.tolist(),
        key="clean_convert_col",
    )
    convert_target = st.selectbox(
        "Convert to",
        options=["numeric", "datetime", "string"],
        key="clean_convert_target",
    )

    if st.button("Apply conversion", key="clean_convert_apply", use_container_width=True):
        before_non_null = int(working_data[convert_col].notna().sum())
        if convert_target == "numeric":
            converted = pd.to_numeric(working_data[convert_col], errors="coerce")
        elif convert_target == "datetime":
            converted = pd.to_datetime(working_data[convert_col], errors="coerce")
        else:
            converted = working_data[convert_col].astype("string")

        after_non_null = int(converted.notna().sum())
        working_data[convert_col] = converted
        st.session_state["cleaning_history"].append(
            f"Converted {convert_col} to {convert_target} ({after_non_null:,}/{before_non_null:,} non-null retained)"
        )

with st.sidebar.expander("Text cleaning", expanded=False):
    current_numeric, current_categorical, current_datetime, _ = column_types(working_data)
    text_cols = st.multiselect(
        "Columns",
        options=current_categorical,
        default=[],
        key="clean_text_cols",
    )
    trim_whitespace = st.checkbox("Trim whitespace", value=True, key="clean_trim_whitespace")
    remove_special = st.checkbox("Remove special characters", value=False, key="clean_remove_special")

    if st.button("Apply text cleaning", key="clean_apply_text", use_container_width=True):
        updated, changed = clean_text_columns(
            working_data,
            text_cols,
            trim_whitespace,
            remove_special,
        )
        working_data = updated
        st.session_state["cleaning_history"].append(
            f"Text cleaning updated {changed:,} cell(s) across {len(text_cols)} column(s)"
        )

working_data = coerce_datetime_columns(working_data)
st.session_state["working_data"] = working_data

data = st.session_state["working_data"]
if data.empty:
    st.error("No rows remain after cleaning actions. Re-upload file or adjust cleaning options.")
    st.stop()

numeric_cols, categorical_cols, datetime_cols, other_cols = column_types(data)
filtered_data, filter_summaries = apply_filters_sidebar(data, numeric_cols, categorical_cols, datetime_cols)

st.subheader("Business Question and Decision Context")
use_case_templates = {
    "Custom": {
        "question": "How can driving behavior and route choices improve Tesla efficiency, battery use, and trip planning?",
        "questions_answered": [
            "Which routes and locations consume the most energy?",
            "How do trip duration and driving patterns vary over time?",
            "Where can charging and trip planning decisions be improved?",
        ],
        "recommended_filters": "Date range, Starting Location, Ending Location, Tag",
    },
    "Route Efficiency Analysis": {
        "question": "Which routes deliver the best and worst energy efficiency, and when do they occur?",
        "questions_answered": [
            "Which start/end routes have highest Wh/mi?",
            "Do long routes differ from short routes in efficiency?",
            "How stable is route efficiency month to month?",
        ],
        "recommended_filters": "Starting Location, Ending Location, Duration (Minutes)",
    },
    "Battery Performance Tracking": {
        "question": "How consistently is battery consumed per mile across trips and periods?",
        "questions_answered": [
            "What is battery drop per 100 miles over time?",
            "Are there outlier trips with abnormal battery consumption?",
            "Which conditions correspond to battery inefficiency?",
        ],
        "recommended_filters": "Started At, Ended At, Distance (mi), Starting Battery (%)",
    },
    "Peak Usage Times": {
        "question": "When are peak driving windows, and what patterns are visible by weekday/weekend?",
        "questions_answered": [
            "Which days and periods have the most trips?",
            "Is weekend behavior different from weekday behavior?",
            "What are the busiest origin/destination combinations?",
        ],
        "recommended_filters": "Started At, Tag, Starting Location",
    },
    "Cost Optimization": {
        "question": "Where can energy consumption be reduced to lower charging cost over time?",
        "questions_answered": [
            "Which trips have the highest kWh use?",
            "Can route choices reduce Wh/mi?",
            "How much can be saved if high-consumption segments are improved?",
        ],
        "recommended_filters": "Total Energy Used (kWh), Average Energy Used (Wh/mi), Distance (mi)",
    },
}
template_choice = st.selectbox(
    "Business use-case template",
    options=list(use_case_templates.keys()),
    index=0,
    key="biz_template_choice",
)
selected_template = use_case_templates[template_choice]
st.markdown(f"This dashboard answers: **{selected_template['question']}**")
st.caption(f"Recommended filters: {selected_template['recommended_filters']}")
st.markdown("**Example questions answered**")
for q in selected_template["questions_answered"]:
    st.write(f"- {q}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Why this matters**")
    st.markdown(
        "- Identify high-energy trips and routes\n"
        "- Improve charging and route planning decisions\n"
        "- Track consistency of driving efficiency over time"
    )
with c2:
    st.markdown("**Business objectives**")
    st.markdown(
        "- Reduce average energy use (Wh/mi)\n"
        "- Monitor battery drop per distance\n"
        "- Identify frequent start/end route clusters"
    )

kpis = calculate_business_kpis(filtered_data)
if kpis:
    kpi_cols = st.columns(len(kpis))
    for idx, (label, value, help_text) in enumerate(kpis):
        kpi_cols[idx].metric(label, value, help=help_text)

full_rows = len(data)
filtered_rows = len(filtered_data)
removed_rows = full_rows - filtered_rows
removed_pct = (removed_rows / full_rows * 100) if full_rows else 0

if filter_summaries:
    st.warning(
        f"Viewing: {filtered_rows:,} of {full_rows:,} rows ({(filtered_rows / full_rows * 100):.1f}%). "
        f"Active filters: {', '.join(filter_summaries)}"
    )
    st.caption(f"Filtering removed {removed_rows:,} rows ({removed_pct:.1f}%).")
else:
    st.info(f"Viewing full cleaned dataset: {full_rows:,} rows. No active filters.")

if filtered_rows == 0:
    st.error("No data matches current filters. Use 'Clear all filters' or loosen filter values.")
elif filtered_rows < max(10, int(0.05 * full_rows)):
    st.warning("Current filters keep less than 5% of rows. Insights may be unstable due to low sample size.")

if st.button("Clear all filters (global)", use_container_width=False):
    clear_filter_state()
    st.rerun()

dup_count = int(data.duplicated().sum())
mem_mb = float(data.memory_usage(deep=True).sum() / (1024 * 1024))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows (cleaned)", f"{len(data):,}")
m2.metric("Rows (filtered)", f"{len(filtered_data):,}")
m3.metric("Columns", f"{data.shape[1]}")
m4.metric("Duplicate rows", f"{dup_count:,}")

st.caption(
    "Loaded using encoding: {enc} (delimiter: {delim}). Memory: {mem:.2f} MB.".format(
        enc=meta.get("encoding", "unknown"),
        delim=meta.get("delimiter", "unknown"),
        mem=mem_mb,
    )
)

if st.session_state["cleaning_history"]:
    with st.expander("Cleaning history", expanded=False):
        for idx, item in enumerate(st.session_state["cleaning_history"], start=1):
            st.write(f"{idx}. {item}")

st.divider()

tab_overview, tab_explore, tab_viz, tab_compare, tab_findings, tab_export, tab_help = st.tabs(
    ["Overview", "Explore", "Visualize", "Compare", "Findings", "Export", "Help"]
)

with tab_overview:
    st.subheader("Data Source and Provenance")
    p1, p2 = st.columns(2)
    with p1:
        st.text_input("Source system", key="src_system")
        st.text_input("Exported on (YYYY-MM-DD)", key="src_export_date")
    with p2:
        st.text_input("Data owner", key="src_owner")
        st.text_input("Source URL/reference", key="src_url")

    st.text_area("Known data quality issues or limitations", key="src_limitations", height=80)

    schema_df = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": data.dtypes.astype(str).values,
            "description": [infer_schema_description(c) for c in data.columns],
        }
    )
    st.dataframe(schema_df, use_container_width=True, height=220)

    st.subheader("Data quality summary")
    left, right = st.columns([1.1, 0.9])

    with left:
        overview = dataset_overview_table(data)
        st.dataframe(overview, use_container_width=True, height=360)

    with right:
        st.subheader("Numeric descriptive stats")
        if numeric_cols:
            stats = numeric_stats(data, numeric_cols)
            st.dataframe(stats, use_container_width=True, height=360)
        else:
            st.info("No numeric columns detected.")

    st.subheader("Missing data heatmap")
    miss_matrix = data.head(500).isna().astype(int).T
    if miss_matrix.empty:
        st.info("No rows available for missing-data heatmap.")
    else:
        miss_fig = px.imshow(
            miss_matrix,
            aspect="auto",
            color_continuous_scale=[[0, "#2a9d8f"], [1, "#e76f51"]],
            labels={"x": "Row index", "y": "Column", "color": "Missing"},
            title="Missing data pattern (first 500 rows)",
        )
        miss_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=420)
        st.plotly_chart(miss_fig, use_container_width=True)

with tab_explore:
    st.subheader("Filtered data preview")
    if filtered_data.empty:
        st.warning("No data matches your filters. Try adjusting filter criteria.")
    else:
        default_cols = filtered_data.columns.tolist()[: min(10, filtered_data.shape[1])]
        show_cols = st.multiselect(
            "Columns to display",
            options=filtered_data.columns.tolist(),
            default=default_cols,
        )
        st.dataframe(filtered_data[show_cols].head(500), use_container_width=True, height=520)

with tab_viz:
    if filtered_data.empty:
        st.warning("No charts available because the filtered dataset is empty.")
    else:
        palette_choice = st.selectbox(
            "Color palette",
            options=["Plotly", "Safe", "Bold", "Dark24", "Set2"],
            index=0,
            key="viz_palette",
        )
        palette_sequence = get_palette_sequence(palette_choice)
        log_histogram = st.toggle("Log scale histogram y-axis", value=False, key="viz_hist_log")
        bar_mode = st.selectbox(
            "Bar layout mode",
            options=["group", "stack"],
            index=0,
            key="viz_bar_mode",
        )
        st.caption(sample_size_label(data, filtered_data, datetime_cols))

        v1, v2 = st.columns(2)

        with v1:
            st.subheader("Histogram (numeric)")
            if numeric_cols:
                hist_col = st.selectbox("Numeric column", options=numeric_cols, index=0, key="viz_hist_col")
                bins = st.slider("Bins", min_value=5, max_value=80, value=30, key="viz_hist_bins")

                color_by = None
                if categorical_cols:
                    color_choices = ["(None)"] + categorical_cols
                    pick = st.selectbox("Color by (optional)", options=color_choices, index=0, key="viz_hist_color")
                    if pick != "(None)":
                        color_by = pick
                hist_title = st.text_input(
                    "Histogram title",
                    value=f"Distribution of {hist_col}",
                    key="viz_hist_title",
                )

                hist_fig = px.histogram(
                    filtered_data,
                    x=hist_col,
                    nbins=bins,
                    color=color_by,
                    title=hist_title,
                    color_discrete_sequence=palette_sequence,
                )
                if log_histogram:
                    hist_fig.update_yaxes(type="log")
                hist_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(hist_fig, use_container_width=True)
                st.caption(sample_size_label(data, filtered_data, datetime_cols))
            else:
                st.info("No numeric columns available for a histogram.")

        with v2:
            st.subheader("Bar chart (categorical)")
            if categorical_cols:
                cat_col = st.selectbox("Categorical column", options=categorical_cols, index=0, key="viz_bar_cat")
                split_options = ["(None)"] + [c for c in categorical_cols if c != cat_col]
                split_by = st.selectbox(
                    "Split by (optional)",
                    options=split_options,
                    index=0,
                    key="viz_bar_split",
                )
                top_n = st.slider("Top categories to show", min_value=5, max_value=30, value=10, key="viz_bar_topn")
                as_percent = st.toggle("Show as percent", value=False, key="viz_bar_pct")
                bar_title = st.text_input(
                    "Bar chart title",
                    value=f"Top {top_n} categories for {cat_col}",
                    key="viz_bar_title",
                )

                if split_by == "(None)":
                    series = filtered_data[cat_col].fillna("(Missing)").astype(str)
                    vc = series.value_counts().head(top_n)
                    bar_df = vc.reset_index()
                    bar_df.columns = [cat_col, "count"]
                    y_field = "count"
                    title = bar_title

                    if as_percent:
                        bar_df["percent"] = bar_df["count"] / bar_df["count"].sum() * 100
                        y_field = "percent"
                        title = f"{bar_title} (%)"

                    bar_fig = px.bar(
                        bar_df,
                        x=cat_col,
                        y=y_field,
                        title=title,
                        color_discrete_sequence=palette_sequence,
                    )
                else:
                    split_df = filtered_data[[cat_col, split_by]].copy()
                    split_df[cat_col] = split_df[cat_col].fillna("(Missing)").astype(str)
                    split_df[split_by] = split_df[split_by].fillna("(Missing)").astype(str)

                    top_categories = split_df[cat_col].value_counts().head(top_n).index
                    split_df = split_df[split_df[cat_col].isin(top_categories)]
                    bar_df = split_df.groupby([cat_col, split_by]).size().reset_index(name="count")
                    y_field = "count"
                    title = bar_title

                    if as_percent:
                        bar_df["percent"] = (
                            bar_df["count"] / bar_df.groupby(cat_col)["count"].transform("sum") * 100
                        )
                        y_field = "percent"
                        title = f"{bar_title} (%)"

                    bar_fig = px.bar(
                        bar_df,
                        x=cat_col,
                        y=y_field,
                        color=split_by,
                        barmode=bar_mode,
                        title=title,
                        color_discrete_sequence=palette_sequence,
                    )

                bar_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                if as_percent:
                    bar_fig.update_yaxes(ticksuffix="%")
                st.plotly_chart(bar_fig, use_container_width=True)
                st.caption(sample_size_label(data, filtered_data, datetime_cols))
            else:
                st.info("No categorical columns available for a bar chart.")

        st.divider()
        st.subheader("Correlation (numeric)")
        if len(numeric_cols) >= 2:
            corr = filtered_data[numeric_cols].corr(numeric_only=True)
            corr_fig = px.imshow(corr, text_auto=".2f", title="Correlation heatmap (filtered view)")
            corr_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(corr_fig, use_container_width=True)
            st.caption(sample_size_label(data, filtered_data, datetime_cols))
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

        st.divider()
        st.subheader("Time series (if dates are available)")
        if datetime_cols and numeric_cols:
            t1, t2, t3 = st.columns([0.38, 0.38, 0.24])
            with t1:
                dt_col = st.selectbox("Date column", options=datetime_cols, index=0)
            with t2:
                val_col = st.selectbox("Value column", options=numeric_cols, index=0)
            with t3:
                freq = st.selectbox("Bucket", options=["D", "W", "M"], index=1)

            agg = st.selectbox("Aggregation", options=["sum", "mean", "median", "count"], index=0)

            ts = filtered_data[[dt_col, val_col]].dropna(subset=[dt_col]).copy()
            if ts.empty:
                st.info("No usable datetime rows after filtering.")
            else:
                ts = ts.set_index(dt_col).sort_index()
                if agg == "count":
                    ts_out = ts[val_col].resample(freq).count().rename("value").reset_index()
                elif agg == "mean":
                    ts_out = ts[val_col].resample(freq).mean().rename("value").reset_index()
                elif agg == "median":
                    ts_out = ts[val_col].resample(freq).median().rename("value").reset_index()
                else:
                    ts_out = ts[val_col].resample(freq).sum().rename("value").reset_index()

                ts_fig = px.line(
                    ts_out,
                    x=dt_col,
                    y="value",
                    title=f"{agg.title()} of {val_col} over time ({freq})",
                    color_discrete_sequence=palette_sequence,
                )
                ts_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(ts_fig, use_container_width=True)
                st.caption(sample_size_label(data, filtered_data, datetime_cols))
        else:
            st.info("Time series requires at least one datetime and one numeric column.")

        st.divider()
        st.subheader("Scatter explorer")
        if len(numeric_cols) >= 2:
            sc1, sc2, sc3 = st.columns([0.4, 0.4, 0.2])
            with sc1:
                x_col = st.selectbox("X-axis", options=numeric_cols, index=0, key="viz_scatter_x")
            with sc2:
                y_candidates = [c for c in numeric_cols if c != x_col] or numeric_cols
                y_col = st.selectbox("Y-axis", options=y_candidates, index=0, key="viz_scatter_y")
            with sc3:
                max_points = st.slider("Max points", min_value=200, max_value=3000, value=1500, step=100, key="viz_scatter_n")

            scatter_df = filtered_data[[x_col, y_col]].dropna().head(max_points)
            if scatter_df.empty:
                st.info("Not enough non-null values for scatter plot.")
            else:
                scatter_fig = px.scatter(
                    scatter_df,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} vs {x_col}",
                    color_discrete_sequence=palette_sequence,
                )
                scatter_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.caption(f"Scatter sample size: n = {len(scatter_df):,}")
        else:
            st.info("Need at least two numeric columns for scatter exploration.")

with tab_compare:
    st.subheader("Comparison and Benchmarking")
    if filtered_data.empty:
        st.warning("No comparison available because filtered data is empty.")
    else:
        st.caption(sample_size_label(data, filtered_data, datetime_cols))

        compare_df = comparison_metrics_table(data, filtered_data, numeric_cols)
        if compare_df.empty:
            st.info("No numeric metrics available for filtered vs full comparison.")
        else:
            metric_options = compare_df["metric"].tolist()
            metric_choice = st.selectbox(
                "Metric to compare",
                options=metric_options,
                index=0,
                key="cmp_metric_choice",
            )
            chosen_row = compare_df[compare_df["metric"] == metric_choice].iloc[0]

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Full-data mean", f"{chosen_row['full_mean']:.3g}")
            cm2.metric("Filtered mean", f"{chosen_row['filtered_mean']:.3g}")
            cm3.metric("Delta vs full", f"{chosen_row['delta']:.3g}", f"{chosen_row['delta_pct']:.1f}%")

            st.dataframe(compare_df, use_container_width=True, height=260)

            comp_fig = px.bar(
                compare_df.head(10),
                x="metric",
                y=["full_mean", "filtered_mean"],
                barmode="group",
                title="Filtered vs full means (top 10 metrics)",
                color_discrete_sequence=get_palette_sequence("Safe"),
            )
            comp_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(comp_fig, use_container_width=True)

        st.divider()
        st.subheader("Time period comparison")
        if datetime_cols and numeric_cols:
            pc1, pc2, pc3 = st.columns([0.4, 0.4, 0.2])
            with pc1:
                period_dt_col = st.selectbox("Date column", options=datetime_cols, index=0, key="cmp_dt_col")
            with pc2:
                period_metric_col = st.selectbox("Metric column", options=numeric_cols, index=0, key="cmp_metric_col")
            with pc3:
                period_freq = st.selectbox("Bucket", options=["W", "M"], index=1, key="cmp_bucket")

            full_ts = data[[period_dt_col, period_metric_col]].dropna().copy()
            filt_ts = filtered_data[[period_dt_col, period_metric_col]].dropna().copy()
            if full_ts.empty or filt_ts.empty:
                st.info("Not enough rows for time period comparison.")
            else:
                full_agg = (
                    full_ts.set_index(period_dt_col)[period_metric_col]
                    .resample(period_freq)
                    .mean()
                    .rename("value")
                    .reset_index()
                )
                full_agg["dataset"] = "Full"

                filt_agg = (
                    filt_ts.set_index(period_dt_col)[period_metric_col]
                    .resample(period_freq)
                    .mean()
                    .rename("value")
                    .reset_index()
                )
                filt_agg["dataset"] = "Filtered"

                comp_time = pd.concat([full_agg, filt_agg], ignore_index=True)
                time_fig = px.line(
                    comp_time,
                    x=period_dt_col,
                    y="value",
                    color="dataset",
                    title=f"Mean {period_metric_col} over time: filtered vs full",
                    color_discrete_sequence=get_palette_sequence("Bold"),
                )
                time_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(time_fig, use_container_width=True)
        else:
            st.info("Need one datetime and one numeric column for time period comparison.")

        st.divider()
        st.subheader("Benchmark target tracking")
        if numeric_cols:
            b1, b2, b3 = st.columns([0.35, 0.35, 0.3])
            with b1:
                benchmark_col = st.selectbox("Benchmark metric", options=numeric_cols, key="cmp_bench_col")
            current_metric = float(filtered_data[benchmark_col].dropna().mean()) if filtered_data[benchmark_col].dropna().shape[0] else 0.0
            with b2:
                benchmark_target = st.number_input(
                    "Target value",
                    value=current_metric,
                    key="cmp_bench_target",
                )
            with b3:
                goal_direction = st.selectbox(
                    "Goal direction",
                    options=["Lower is better", "Higher is better"],
                    key="cmp_goal_direction",
                )

            if goal_direction == "Lower is better":
                met_target = current_metric <= benchmark_target
                gap = current_metric - benchmark_target
                status_line = f"Current mean {benchmark_col}: {current_metric:.3g}. Gap to target: {gap:.3g}."
            else:
                met_target = current_metric >= benchmark_target
                gap = benchmark_target - current_metric
                status_line = f"Current mean {benchmark_col}: {current_metric:.3g}. Gap to target: {gap:.3g}."

            if met_target:
                st.success(f"Target met. {status_line}")
            else:
                st.warning(f"Target not yet met. {status_line}")
        else:
            st.info("No numeric columns available for benchmark tracking.")

with tab_findings:
    st.subheader("Top Insights")
    insights = generate_findings(data, filtered_data, numeric_cols, categorical_cols, datetime_cols)
    for idx, insight in enumerate(insights, start=1):
        st.write(f"{idx}. {insight}")
    st.caption("Insights are heuristic and intended to accelerate exploratory analysis.")

with tab_export:
    st.subheader("Export")
    if filtered_data.empty:
        st.warning("Filtered dataset is empty, so there is nothing to export.")
    else:
        csv_bytes = filtered_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_data.csv",
            mime="text/csv",
        )

        source_info = {
            "source_system": st.session_state.get("src_system", ""),
            "export_date": st.session_state.get("src_export_date", ""),
            "owner": st.session_state.get("src_owner", ""),
            "source_url": st.session_state.get("src_url", ""),
            "limitations": st.session_state.get("src_limitations", ""),
        }

        insights = generate_findings(data, filtered_data, numeric_cols, categorical_cols, datetime_cols)
        summary = build_summary_payload(
            data,
            filtered_data,
            filter_summaries,
            insights,
            meta,
            source_info,
            datetime_cols,
        )

        summary_json = json.dumps(summary, indent=2)
        summary_html = build_html_report(summary)

        st.download_button(
            label="Download analysis summary (JSON)",
            data=summary_json,
            file_name="analysis_summary.json",
            mime="application/json",
        )
        st.download_button(
            label="Download analysis summary (HTML)",
            data=summary_html,
            file_name="analysis_summary.html",
            mime="text/html",
        )
        st.download_button(
            label="Download README template",
            data=build_readme_text(),
            file_name="README.md",
            mime="text/markdown",
        )

        st.caption(
            f"Export includes {len(filtered_data):,} rows, {filtered_data.shape[1]} columns, and a narrative summary."
        )

with tab_help:
    st.subheader("How to use this dashboard")
    st.markdown(
        "1. Upload CSV.\n"
        "2. Optionally clean data from the sidebar.\n"
        "3. Apply filters for categorical, numeric, and date columns.\n"
        "4. Review KPIs, visualizations, findings, and comparisons.\n"
        "5. Export filtered data and summary artifacts."
    )
    st.markdown("**Keyboard-friendly tips**")
    st.write("- Press `/` then type widget labels to jump quickly in browser search.")
    st.write("- Use the clear-filter buttons when no rows match.")
    st.write("- Keep filters narrow for focused insights, broad for trend analysis.")

    st.markdown("**Expected input**")
    st.write("- CSV with headers")
    st.write("- Mixed data types are supported")
    st.write("- Datetime columns are auto-detected when parse confidence is high")

    st.markdown("**Troubleshooting**")
    st.write("- Upload errors: verify file is CSV and below 200MB")
    st.write("- Empty charts: verify required numeric/categorical columns exist")
    st.write("- Unexpected results: review cleaning history and active filters")

    st.code(build_readme_text(), language="markdown")
