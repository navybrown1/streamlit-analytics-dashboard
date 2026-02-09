from __future__ import annotations

import functools
import io
import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# AI / ML imports (safe fallbacks)
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# =============================================================================
# 1. CONFIGURATION & THEME
# =============================================================================

MAX_UPLOAD_MB = 200

st.set_page_config(
    page_title="Business Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, professional UI theme with CSS variables
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-hover: #4f46e5;
        --secondary: #8b5cf6;
        --bg: #f8fafc;
        --card-bg: #ffffff;
        --text: #1e293b;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --sidebar-bg: #0f172a;
        --sidebar-text: #94a3b8;
        --sidebar-heading: #f8fafc;
        --success: #10b981;
        --warning: #f59e0b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stCaption {
        color: var(--sidebar-text) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--sidebar-heading) !important;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, var(--primary-hover), #7c3aed);
        box-shadow: 0 6px 10px -1px rgba(99, 102, 241, 0.4);
    }

    /* --- HEADER BANNER --- */
    .dashboard-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.4);
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('https://www.transparenttextures.com/patterns/cubes.png');
        opacity: 0.08;
    }
    .dashboard-header h1 {
        color: white !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.5rem 0 !important;
        position: relative;
    }
    .dashboard-header p {
        color: #e0e7ff !important;
        font-size: 1.05rem;
        max-width: 640px;
        position: relative;
    }

    /* --- METRICS --- */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid var(--border-light);
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.12);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }

    /* --- BUTTONS (main area) --- */
    .stButton > button {
        background: white;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        border-color: var(--primary);
        color: var(--primary);
        background: #f5f3ff;
    }

    /* --- DOWNLOAD BUTTONS --- */
    .stDownloadButton > button {
        background: linear-gradient(90deg, var(--success), #059669);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 6px 12px -1px rgba(16, 185, 129, 0.4);
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: var(--text-secondary);
        border: none;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background-color: #f5f3ff;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* --- EXPANDERS --- */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--text);
        background-color: white;
        border-radius: 8px;
    }

    /* --- CARDS --- */
    .ui-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .ui-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }

    /* --- FILE UPLOADER --- */
    section[data-testid="stFileUploadDropzone"] {
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
        background: white;
        transition: all 0.2s;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--primary);
        background: #f5f3ff;
    }

    /* --- DATAFRAMES --- */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* --- MISC --- */
    .block-container { padding-top: 2rem; }
    hr { border: none; height: 1px; background: var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

# =============================================================================
# 2a. AI / ML HELPER FUNCTIONS
# =============================================================================

def get_gemini_key() -> str | None:
    """Resolve Gemini API key from secrets, env, or session state."""
    # 1. Streamlit secrets (Streamlit Cloud)
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 2. Environment variable
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        return key
    # 3. Session state (sidebar input)
    return st.session_state.get("user_gemini_key", "")


def init_gemini(api_key: str):
    """Configure Gemini and return a working model (tries multiple models)."""
    genai.configure(api_key=api_key)

    # Try models in order: newest stable free-tier first, then fallbacks
    model_candidates = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-3-flash-preview",
    ]

    # If we already found a working model in this session, reuse it
    cached = st.session_state.get("_gemini_model_name")
    if cached:
        return genai.GenerativeModel(cached)

    # Probe each model with a tiny request
    for name in model_candidates:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("Hi", generation_config={"max_output_tokens": 5})
            st.session_state["_gemini_model_name"] = name
            return model
        except Exception:
            continue

    # Last resort: just return the first candidate and let the caller handle errors
    return genai.GenerativeModel(model_candidates[0])


def build_gemini_context(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
    filter_summaries: list[str],
    kpis: list[tuple[str, str, str]],
) -> str:
    """Build a concise data context string for the LLM (no raw data sent)."""
    lines: list[str] = []
    lines.append(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns.")
    lines.append(f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'none'}")
    lines.append(f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'none'}")
    lines.append(f"Datetime columns: {', '.join(datetime_cols) if datetime_cols else 'none'}")

    if filter_summaries:
        lines.append(f"Active filters: {'; '.join(filter_summaries)}")
    else:
        lines.append("No filters active — viewing full dataset.")

    # Summary stats
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2).to_string()
        lines.append(f"\nNumeric summary:\n{desc}")

    # KPIs
    if kpis:
        lines.append("\nKey Performance Indicators:")
        for label, value, helptext in kpis:
            lines.append(f"  - {label}: {value} ({helptext})")

    # Sample rows (max 5)
    sample = df.head(5).to_string(index=False)
    lines.append(f"\nSample rows (first 5):\n{sample}")

    # Missing data summary
    miss = df.isna().sum()
    miss_nonzero = miss[miss > 0]
    if not miss_nonzero.empty:
        lines.append(f"\nMissing values: {miss_nonzero.to_dict()}")

    return "\n".join(lines)


def run_anomaly_detection(
    df: pd.DataFrame, columns: list[str], contamination: float = 0.05
) -> tuple[pd.DataFrame, int]:
    """Run Isolation Forest on selected columns. Returns df with anomaly flag."""
    subset = df[columns].dropna()
    if subset.shape[0] < 10:
        return df.assign(_anomaly=False), 0

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(subset)
    # -1 = anomaly, 1 = normal
    anomaly_flags = pd.Series(False, index=df.index)
    anomaly_flags.loc[subset.index] = preds == -1

    result = df.copy()
    result["_anomaly"] = anomaly_flags
    anomaly_count = int(anomaly_flags.sum())
    return result, anomaly_count


def forecast_trend(
    df: pd.DataFrame, dt_col: str, val_col: str, periods: int = 12, degree: int = 1
) -> tuple[pd.DataFrame, float]:
    """
    Fit polynomial regression on time series and forecast ahead.
    Returns forecast DataFrame and R-squared score.
    """
    ts = df[[dt_col, val_col]].dropna().sort_values(dt_col).copy()
    if ts.shape[0] < 5:
        return pd.DataFrame(), 0.0

    # Convert dates to numeric (days since first date)
    ts["_days"] = (ts[dt_col] - ts[dt_col].min()).dt.total_seconds() / 86400.0
    x = ts["_days"].values
    y = ts[val_col].values

    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)

    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual std for confidence band
    residual_std = float(np.std(y - y_pred))

    # Historical fitted
    hist = pd.DataFrame({
        dt_col: ts[dt_col].values,
        "actual": y,
        "fitted": y_pred,
        "type": "Historical",
    })

    # Future forecast
    last_date = ts[dt_col].max()
    avg_gap = np.median(np.diff(x)) if len(x) > 1 else 1.0
    future_days = np.array([x[-1] + avg_gap * (i + 1) for i in range(periods)])
    future_dates = [last_date + pd.Timedelta(days=float(d - x[-1])) for d in future_days]
    future_vals = poly(future_days)

    forecast = pd.DataFrame({
        dt_col: future_dates,
        "actual": [np.nan] * periods,
        "fitted": future_vals,
        "type": "Forecast",
    })

    combined = pd.concat([hist, forecast], ignore_index=True)

    # Widening confidence band: uncertainty grows with forecast horizon
    horizon = np.zeros(len(combined))
    n_hist = len(hist)
    n_total = len(combined)
    for i in range(n_hist, n_total):
        horizon[i] = i - n_hist + 1
    band_width = 1.96 * residual_std * np.sqrt(1 + horizon / max(1, n_hist))
    combined["lower"] = combined["fitted"] - band_width
    combined["upper"] = combined["fitted"] + band_width

    return combined, r_squared


def update_chart_design(fig, height=400):
    """Apply consistent clean Plotly styling."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#1e293b"),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=16, color="#0f172a", family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
    return fig


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

    # Heuristic: reclassify low-cardinality numeric columns as categorical.
    # Columns like "Tag" often hold values 1,2,3 that pandas reads as int
    # but are really labels, not measurements.
    reclassified: list[str] = []
    for col in numeric_cols:
        nunique = df[col].nunique(dropna=True)
        n_rows = len(df)
        # Few unique values AND column name hints at a label (or <= 12 distinct values
        # in a dataset with > 20 rows → almost certainly categorical)
        is_low_card = nunique <= 12 and n_rows > 20
        name_hint = any(
            kw in col.lower()
            for kw in ["tag", "id", "code", "flag", "type", "class", "category", "label", "group", "level", "status"]
        )
        if is_low_card or name_hint:
            reclassified.append(col)
    for col in reclassified:
        numeric_cols.remove(col)
        categorical_cols.append(col)

    other_cols = [
        col for col in df.columns
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

    overview = pd.DataFrame({
        "Column": df.columns,
        "Data Type": dtype.values,
        "Non-Null Count": (df.shape[0] - missing).values,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
        "Unique Values": unique_values.values,
        "Suggested Fix": rec,
    })
    return overview.sort_values(["Missing %", "Unique Values"], ascending=[False, False])


def numeric_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["Column", "Mean", "Median", "Std Dev", "Min", "Max"])
    data = df[numeric_cols]
    stats = pd.DataFrame({
        "Mean": data.mean(numeric_only=True),
        "Median": data.median(numeric_only=True),
        "Std Dev": data.std(numeric_only=True),
        "Min": data.min(numeric_only=True),
        "Max": data.max(numeric_only=True),
    })
    return stats.reset_index().rename(columns={"index": "Column"})


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
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+\d{4}",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        try:
            return datetime.strptime(month_match.group(0), "%B %d %Y").date().isoformat()
        except Exception:
            return datetime.now().date().isoformat()

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
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
        kpis.append((
            "Avg Trip Duration",
            f"{df[duration_col].dropna().mean():.1f} min",
            "Average trip time across filtered dataset",
        ))

    if avg_energy_col and pd.api.types.is_numeric_dtype(df[avg_energy_col]):
        # Filter to plausible efficiency values (0-1200 Wh/mi) and trips > 0.2 mi
        eff_series = df[avg_energy_col].dropna()
        if distance_col and pd.api.types.is_numeric_dtype(df[distance_col]):
            dist_mask = df[distance_col].fillna(0) > 0.2
            eff_series = df.loc[dist_mask, avg_energy_col].dropna()
        eff_series = eff_series[eff_series.between(0, 1200)]
        if not eff_series.empty:
            kpis.append((
                "Avg Trip Efficiency",
                f"{eff_series.median():.1f} Wh/mi",
                "Median Wh/mi (excludes tiny trips and extreme outliers)",
            ))

    if (
        start_battery_col and end_battery_col and distance_col
        and pd.api.types.is_numeric_dtype(df[start_battery_col])
        and pd.api.types.is_numeric_dtype(df[end_battery_col])
        and pd.api.types.is_numeric_dtype(df[distance_col])
    ):
        subset = df[[start_battery_col, end_battery_col, distance_col]].dropna().copy()
        # Only use trips > 1 mile — short trips produce wildly inflated per-mile rates
        subset = subset[subset[distance_col] > 1.0]
        # Remove impossible battery values
        subset = subset[
            subset[start_battery_col].between(0, 100)
            & subset[end_battery_col].between(0, 100)
        ]
        if not subset.empty:
            drop_pct = subset[start_battery_col] - subset[end_battery_col]
            rate = (drop_pct / subset[distance_col]) * 100
            # Remove negative rates (battery gained during trip = regen/charging artifact)
            rate = rate[rate >= 0]
            if not rate.dropna().empty:
                kpis.append((
                    "Battery Drop / 100 mi",
                    f"{rate.dropna().median():.2f}%",
                    "Median battery % per 100 mi (trips > 1 mi, excludes anomalies)",
                ))

    if start_col and end_col:
        route_series = (
            df[start_col].fillna("(Missing)").astype(str)
            + " → "
            + df[end_col].fillna("(Missing)").astype(str)
        )
        if not route_series.empty:
            top_route = route_series.value_counts().head(1)
            if not top_route.empty:
                kpis.append((
                    "Most Frequent Route",
                    top_route.index[0][:24],
                    f"{int(top_route.iloc[0]):,} trip(s) for top route",
                ))

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
            series = series.str.replace(r"[^0-9a-zA-Z\s\-\.,:/()&]", "", regex=True)
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
        rows.append({
            "metric": col,
            "full_mean": full_mean,
            "filtered_mean": filtered_mean,
            "delta": delta,
            "delta_pct": delta_pct,
        })
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
- If filters return no rows, click "Clear all filters".
- If charts are empty, verify at least one numeric/categorical column exists.
"""


def apply_filters_sidebar(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    st.sidebar.markdown("## 🔍 Filters")
    st.sidebar.caption("Refine your dataset. All visuals update automatically.")

    if st.sidebar.button("Reset all filters", use_container_width=True):
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
        st.sidebar.markdown(f"**{col}**")

        if col in categorical_cols:
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
                f"Trip duration: average **{avg_dur:.1f} min**, std dev **{std_dur:.1f}**, "
                f"95% of trips under **{p95_dur:.1f} min**."
            )
            insights.append(
                f"Trip mix: **{short_pct:.1f}%** short (<10 min) vs **{long_pct:.1f}%** long (>30 min)."
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
                            f"Energy efficiency trend appears **{trend}** from "
                            f"**{monthly.index[0].date()}** to **{monthly.index[-1].date()}**."
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
            "Scatter explorer",
            "Missing data heatmap",
        ],
    }


def build_html_report(summary: dict) -> str:
    filters_html = (
        "<li>No active filters</li>"
        if not summary["filters"]
        else "".join(f"<li>{item}</li>" for item in summary["filters"])
    )
    insights_html = "".join(f"<li>{item}</li>" for item in summary["insights"])
    visuals_html = "".join(f"<li>{item}</li>" for item in summary["visualizations"])

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Analytics Summary Report</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
    h1 {{ color: #1e293b; border-bottom: 2px solid #6366f1; padding-bottom: 8px; }}
    h2 {{ color: #334155; margin-top: 24px; }}
    .meta {{ background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; }}
    li {{ margin-bottom: 4px; }}
  </style>
</head>
<body>
  <h1>Analytics Summary Report</h1>
  <p>Generated: {summary['generated_at']}</p>
  <div class="meta">
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

  <h2>Visualizations Generated</h2>
  <ul>{visuals_html}</ul>
</body>
</html>
""".strip()


# =============================================================================
# 3. MAIN APPLICATION LAYOUT
# =============================================================================

# --- Hero Header ---
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Business Analytics Dashboard</h1>
    <p>Interactive exploration, data quality diagnostics, and business insights from your CSV data.</p>
</div>
""", unsafe_allow_html=True)

# --- File Upload ---
col_upload_1, col_upload_2 = st.columns([1, 2])
with col_upload_1:
    st.markdown("### 📂 Data Import")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    session_file = st.file_uploader(
        "Restore a previous session (optional)",
        type=["json"],
        key="session_restore_uploader",
        help="Upload a dashboard_session.json file to restore cleaning history, business context, and AI chat.",
    )

if not uploaded_files:
    with col_upload_2:
        st.info(
            "👋 **Welcome!** Upload one or more CSV files to begin. "
            "The dashboard auto-detects schema, data types, and datetime columns. "
            "Upload multiple files to stack or merge them."
        )
    st.stop()

# --- Parse each uploaded CSV ---
parsed_frames: list[pd.DataFrame] = []
parsed_metas: list[dict] = []
file_names: list[str] = []
for uf in uploaded_files:
    if uf.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"**{uf.name}** exceeds {MAX_UPLOAD_MB}MB limit. Please upload a smaller CSV.")
        st.stop()
    try:
        raw_bytes = uf.getvalue()
        df_i, meta_i = read_csv_bytes(raw_bytes)
    except Exception as exc:
        st.error(f"Could not read **{uf.name}** as a CSV. Check delimiter, encoding, and file format.")
        st.exception(exc)
        st.stop()
    if df_i is None or df_i.shape[1] == 0:
        st.error(f"**{uf.name}** has no columns.")
        st.stop()
    if df_i.empty:
        st.error(f"**{uf.name}** has no rows.")
        st.stop()
    df_i = df_i.dropna(how="all").copy()
    if df_i.empty:
        st.error(f"**{uf.name}** has no usable rows after removing empty rows.")
        st.stop()
    parsed_frames.append(df_i)
    parsed_metas.append(meta_i)
    file_names.append(uf.name)

# --- Combine mode (only when 2+ files) ---
if len(parsed_frames) == 1:
    data_raw = parsed_frames[0]
    meta = parsed_metas[0]
else:
    with col_upload_2:
        st.markdown(f"**{len(parsed_frames)} files uploaded:** {', '.join(file_names)}")
        combine_mode = st.radio(
            "How should these files be combined?",
            options=["Stack rows (same or similar columns)", "Merge on a shared column"],
            key="csv_combine_mode",
            horizontal=True,
        )
        if combine_mode.startswith("Stack"):
            data_raw = pd.concat(parsed_frames, ignore_index=True)
            st.success(
                f"Stacked {len(parsed_frames)} files → {data_raw.shape[0]:,} rows, "
                f"{data_raw.shape[1]} columns"
            )
        else:
            # Find columns shared across ALL DataFrames
            shared_cols = list(functools.reduce(
                lambda a, b: a & b,
                (set(df.columns) for df in parsed_frames),
            ))
            if not shared_cols:
                st.error("No columns in common across all files. Use 'Stack rows' instead.")
                st.stop()
            merge_key = st.selectbox(
                "Merge key (column present in all files)",
                options=sorted(shared_cols),
                key="csv_merge_key",
            )
            data_raw = parsed_frames[0]
            for i in range(1, len(parsed_frames)):
                data_raw = pd.merge(
                    data_raw, parsed_frames[i],
                    on=merge_key, how="outer",
                    suffixes=("", f"_{file_names[i]}"),
                )
            st.success(
                f"Merged on **{merge_key}** → {data_raw.shape[0]:,} rows, "
                f"{data_raw.shape[1]} columns"
            )
        st.caption(f"Combined columns: {', '.join(data_raw.columns[:20])}{'...' if data_raw.shape[1] > 20 else ''}")
    meta = parsed_metas[0]  # Use first file's encoding/delimiter metadata

data_raw, renamed_columns = make_unique_columns(data_raw)
data_raw = coerce_datetime_columns(data_raw)

# --- Session State Init ---
dataset_token = "|".join(f"{uf.name}:{uf.size}" for uf in uploaded_files)
if st.session_state.get("dataset_token") != dataset_token:
    st.session_state["dataset_token"] = dataset_token
    st.session_state["working_data"] = data_raw.copy()
    st.session_state["cleaning_history"] = []

    has_tesla = any("tesla" in fn.lower() for fn in file_names)
    default_source = "Tesla app CSV export" if has_tesla else "Uploaded CSV file"
    st.session_state["src_system"] = default_source
    st.session_state["src_export_date"] = guess_export_date(file_names[0])
    st.session_state["src_owner"] = "Edwin Brown"
    st.session_state["src_url"] = "Local file upload"
    st.session_state["src_limitations"] = "Route names and tags may be user-entered and inconsistent."

if renamed_columns:
    st.toast(f"Renamed {len(renamed_columns)} duplicate column(s) to avoid ambiguity", icon="⚠️")

# --- Restore session from JSON (if provided) ---
if session_file is not None:
    _restore_token = f"session_restored_{session_file.name}:{session_file.size}"
    if st.session_state.get("_session_restore_token") != _restore_token:
        try:
            _session_data = json.loads(session_file.getvalue().decode("utf-8"))
            _RESTORABLE = [
                "cleaning_history", "src_system", "src_export_date",
                "src_owner", "src_url", "src_limitations", "ai_chat_history",
            ]
            restored_count = 0
            for _rk in _RESTORABLE:
                if _rk in _session_data:
                    st.session_state[_rk] = _session_data[_rk]
                    restored_count += 1
            st.session_state["_session_restore_token"] = _restore_token
            saved_at = _session_data.get("_saved_at", "unknown time")
            st.toast(f"Session restored ({restored_count} fields from {saved_at})", icon="✅")
        except Exception as _restore_err:
            st.warning(f"Could not restore session: {_restore_err}")

# =============================================================================
# 4. SIDEBAR: AI CONFIG + CLEANING TOOLS
# =============================================================================

st.sidebar.markdown("## 🤖 AI Configuration")
if HAS_GEMINI:
    _existing_key = get_gemini_key()
    if _existing_key:
        st.sidebar.success("Gemini API key detected", icon="✅")
    else:
        _user_key = st.sidebar.text_input(
            "Google Gemini API key",
            type="password",
            key="sidebar_gemini_key_input",
            help="Get a free key at https://aistudio.google.com/apikey",
        )
        if _user_key:
            st.session_state["user_gemini_key"] = _user_key
            st.rerun()
else:
    st.sidebar.caption("Install `google-generativeai` to enable AI chat & reports.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧹 Data Cleaning")
st.sidebar.caption("Apply optional cleanup before filtering and visualization.")
working_data = st.session_state["working_data"]

if st.sidebar.button("Drop rows with missing values", use_container_width=True):
    before = len(working_data)
    working_data = working_data.dropna().copy()
    removed = before - len(working_data)
    st.session_state["cleaning_history"].append(
        f"Dropped missing rows: {before:,} → {len(working_data):,} (removed {removed:,})"
    )

if st.sidebar.button("Remove duplicate rows", use_container_width=True):
    before = len(working_data)
    working_data = working_data.drop_duplicates().copy()
    removed = before - len(working_data)
    st.session_state["cleaning_history"].append(
        f"Removed duplicates: {before:,} → {len(working_data):,} (removed {removed:,})"
    )

with st.sidebar.expander("Remove outliers (IQR)", expanded=False):
    st.caption("Removes extreme values using the Interquartile Range method. Works on any numeric column.")
    _iqr_num_cols = working_data.select_dtypes(include=["number"]).columns.tolist()
    iqr_cols = st.multiselect(
        "Columns to check",
        options=_iqr_num_cols,
        default=[],
        key="clean_iqr_cols",
    )
    iqr_factor = st.slider(
        "Sensitivity (IQR multiplier)",
        min_value=1.0, max_value=3.0, value=1.5, step=0.25,
        key="clean_iqr_factor",
        help="1.5 = standard (removes obvious outliers). Lower = more aggressive. Higher = more lenient.",
    )
    if st.button("Remove outliers", key="clean_apply_iqr", use_container_width=True):
        before = len(working_data)
        mask = pd.Series(True, index=working_data.index)
        for col in iqr_cols:
            q1 = working_data[col].quantile(0.25)
            q3 = working_data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            mask = mask & (working_data[col].between(lower, upper) | working_data[col].isna())
        working_data = working_data[mask].copy()
        removed = before - len(working_data)
        st.session_state["cleaning_history"].append(
            f"IQR outlier removal ({', '.join(iqr_cols)}): {before:,} → {len(working_data):,} (removed {removed:,})"
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

    if st.button("Apply fill", key="clean_apply_fill", use_container_width=True):
        updated, filled_cells = apply_fill_missing(
            working_data, numeric_strategy, categorical_strategy, numeric_custom, categorical_custom,
        )
        working_data = updated
        st.session_state["cleaning_history"].append(f"Filled missing values: {filled_cells:,} cell(s)")

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
            f"Converted {convert_col} → {convert_target} ({after_non_null:,}/{before_non_null:,} non-null)"
        )

with st.sidebar.expander("Text cleaning", expanded=False):
    current_numeric, current_categorical, current_datetime, _ = column_types(working_data)
    text_cols = st.multiselect("Columns", options=current_categorical, default=[], key="clean_text_cols")
    trim_whitespace = st.checkbox("Trim whitespace", value=True, key="clean_trim_whitespace")
    remove_special = st.checkbox("Remove special characters", value=False, key="clean_remove_special")

    if st.button("Apply text cleaning", key="clean_apply_text", use_container_width=True):
        updated, changed = clean_text_columns(working_data, text_cols, trim_whitespace, remove_special)
        working_data = updated
        st.session_state["cleaning_history"].append(
            f"Text cleaning: {changed:,} cell(s) across {len(text_cols)} column(s)"
        )

working_data = coerce_datetime_columns(working_data)
st.session_state["working_data"] = working_data

# =============================================================================
# 5. DATA PIPELINE
# =============================================================================

data = st.session_state["working_data"]
if data.empty:
    st.error("No rows remain after cleaning. Re-upload or adjust cleaning options.")
    st.stop()

numeric_cols, categorical_cols, datetime_cols, other_cols = column_types(data)
filtered_data, filter_summaries = apply_filters_sidebar(data, numeric_cols, categorical_cols, datetime_cols)

# --- Business Context ---
st.markdown("### 🎯 Business Question and Decision Context")

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

col_ctx1, col_ctx2 = st.columns([1, 2])
with col_ctx1:
    template_choice = st.selectbox(
        "Analysis goal",
        options=list(use_case_templates.keys()),
        index=0,
        key="biz_template_choice",
    )
    selected_template = use_case_templates[template_choice]

with col_ctx2:
    st.info(f"**Question:** {selected_template['question']}\n\n**Focus:** {selected_template['recommended_filters']}")

with st.expander("Example questions answered & objectives", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📋 Questions answered**")
        for q in selected_template["questions_answered"]:
            st.write(f"• {q}")
    with c2:
        st.markdown("**🎯 Business objectives**")
        st.markdown(
            "• Reduce average energy use (Wh/mi)\n"
            "• Monitor battery drop per distance\n"
            "• Identify frequent start/end route clusters"
        )

# --- KPIs ---
st.markdown("### 📈 Key Performance Indicators")
kpis = calculate_business_kpis(filtered_data)
if kpis:
    kpi_cols = st.columns(len(kpis))
    for idx, (label, value, help_text) in enumerate(kpis):
        kpi_cols[idx].metric(label, value, help=help_text)
else:
    st.info("Insufficient data to generate KPIs.")

# --- Filter status ---
full_rows = len(data)
filtered_rows = len(filtered_data)
removed_rows = full_rows - filtered_rows
removed_pct = (removed_rows / full_rows * 100) if full_rows else 0

if filter_summaries:
    st.warning(
        f"🔍 Viewing {filtered_rows:,} of {full_rows:,} rows ({(filtered_rows / full_rows * 100):.1f}%). "
        f"Active filters: {', '.join(filter_summaries)}"
    )
else:
    st.info(f"Viewing full cleaned dataset: {full_rows:,} rows. No active filters.")

if filtered_rows == 0:
    st.error("No data matches current filters. Use 'Reset all filters' or loosen criteria.")
elif filtered_rows < max(10, int(0.05 * full_rows)):
    st.warning("Current filters keep less than 5% of rows. Insights may be unstable.")

# --- Data stats (collapsed) ---
with st.expander("📊 Data Stats & Cleaning History", expanded=False):
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows (cleaned)", f"{len(data):,}")
    s2.metric("Rows (filtered)", f"{len(filtered_data):,}")
    s3.metric("Columns", f"{data.shape[1]}")
    s4.metric("Duplicates", f"{int(data.duplicated().sum()):,}")

    mem_mb = float(data.memory_usage(deep=True).sum() / (1024 * 1024))
    st.caption(
        f"Encoding: {meta.get('encoding', 'unknown')} | "
        f"Delimiter: {meta.get('delimiter', 'unknown')} | "
        f"Memory: {mem_mb:.2f} MB"
    )

    if st.session_state["cleaning_history"]:
        st.markdown("---")
        st.markdown("**Cleaning history:**")
        for idx, item in enumerate(st.session_state["cleaning_history"], start=1):
            st.caption(f"{idx}. {item}")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 6. TABS
# =============================================================================

tab_overview, tab_explore, tab_viz, tab_compare, tab_findings, tab_ai, tab_export, tab_help = st.tabs(
    ["📋 Overview", "🔎 Explore", "📊 Visualize", "⚖️ Compare", "💡 Findings", "🤖 AI Analysis", "📥 Export", "❓ Help"]
)

# ---------- OVERVIEW ----------
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

    col_ov1, col_ov2 = st.columns([3, 2])
    with col_ov1:
        st.subheader("Data Quality Summary")
        overview = dataset_overview_table(data)
        st.dataframe(overview, use_container_width=True, height=400)
    with col_ov2:
        st.subheader("Numeric Descriptive Stats")
        if numeric_cols:
            stats = numeric_stats(data, numeric_cols)
            st.dataframe(stats, use_container_width=True, height=400)
        else:
            st.info("No numeric columns detected.")

    st.subheader("Column Schema")
    schema_df = pd.DataFrame({
        "column": data.columns,
        "dtype": data.dtypes.astype(str).values,
        "description": [infer_schema_description(c) for c in data.columns],
    })
    st.dataframe(schema_df, use_container_width=True, height=220)

    st.subheader("Missing Data Heatmap")
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
        st.plotly_chart(update_chart_design(miss_fig, height=350), use_container_width=True)

# ---------- EXPLORE ----------
with tab_explore:
    st.subheader("Filtered Data Preview")
    if filtered_data.empty:
        st.warning("Filters too strict — no data to show.")
    else:
        default_cols = filtered_data.columns.tolist()[: min(10, filtered_data.shape[1])]
        show_cols = st.multiselect("Columns to display", options=filtered_data.columns.tolist(), default=default_cols)
        st.caption(f"Showing top 500 of {len(filtered_data):,} rows")
        st.dataframe(filtered_data[show_cols].head(500), use_container_width=True, height=560)

# ---------- VISUALIZE ----------
with tab_viz:
    if filtered_data.empty:
        st.warning("No charts available — filtered dataset is empty.")
    else:
        # Controls row
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            palette_choice = st.selectbox("Color palette", ["Plotly", "Safe", "Bold", "Dark24", "Set2"], key="viz_palette")
        with vc2:
            log_histogram = st.toggle("Log-scale histogram", value=False, key="viz_hist_log")
        with vc3:
            bar_mode = st.selectbox("Bar layout", ["group", "stack"], key="viz_bar_mode")

        palette_sequence = get_palette_sequence(palette_choice)
        st.caption(sample_size_label(data, filtered_data, datetime_cols))

        # --- Histogram + Bar side by side ---
        v1, v2 = st.columns(2)

        with v1:
            st.subheader("Histogram (numeric)")
            if numeric_cols:
                hist_col = st.selectbox("Numeric column", options=numeric_cols, index=0, key="viz_hist_col")
                bins = st.slider("Bins", 5, 80, 30, key="viz_hist_bins")

                color_by = None
                if categorical_cols:
                    color_choices = ["(None)"] + categorical_cols
                    pick = st.selectbox("Color by", options=color_choices, index=0, key="viz_hist_color")
                    if pick != "(None)":
                        color_by = pick

                hist_fig = px.histogram(
                    filtered_data, x=hist_col, nbins=bins, color=color_by,
                    title=f"Distribution of {hist_col}",
                    color_discrete_sequence=palette_sequence,
                )
                if log_histogram:
                    hist_fig.update_yaxes(type="log")
                st.plotly_chart(update_chart_design(hist_fig), use_container_width=True)
            else:
                st.info("No numeric columns available.")

        with v2:
            st.subheader("Bar Chart (categorical)")
            if categorical_cols:
                cat_col = st.selectbox("Categorical column", options=categorical_cols, index=0, key="viz_bar_cat")
                split_options = ["(None)"] + [c for c in categorical_cols if c != cat_col]
                split_by = st.selectbox("Split by", options=split_options, index=0, key="viz_bar_split")
                top_n = st.slider("Top categories", 5, 30, 10, key="viz_bar_topn")
                as_percent = st.toggle("Show as percent", value=False, key="viz_bar_pct")

                if split_by == "(None)":
                    series = filtered_data[cat_col].fillna("(Missing)").astype(str)
                    vc = series.value_counts().head(top_n)
                    bar_df = vc.reset_index()
                    bar_df.columns = [cat_col, "count"]
                    y_field = "count"
                    title = f"Top {top_n}: {cat_col}"

                    if as_percent:
                        bar_df["percent"] = bar_df["count"] / bar_df["count"].sum() * 100
                        y_field = "percent"
                        title += " (%)"

                    bar_fig = px.bar(
                        bar_df, x=cat_col, y=y_field, title=title,
                        color=y_field, color_continuous_scale="Bluyl",
                    )
                else:
                    split_df = filtered_data[[cat_col, split_by]].copy()
                    split_df[cat_col] = split_df[cat_col].fillna("(Missing)").astype(str)
                    split_df[split_by] = split_df[split_by].fillna("(Missing)").astype(str)

                    top_categories = split_df[cat_col].value_counts().head(top_n).index
                    split_df = split_df[split_df[cat_col].isin(top_categories)]
                    bar_df = split_df.groupby([cat_col, split_by]).size().reset_index(name="count")
                    y_field = "count"
                    title = f"Top {top_n}: {cat_col}"

                    if as_percent:
                        bar_df["percent"] = (
                            bar_df["count"] / bar_df.groupby(cat_col)["count"].transform("sum") * 100
                        )
                        y_field = "percent"
                        title += " (%)"

                    bar_fig = px.bar(
                        bar_df, x=cat_col, y=y_field, color=split_by,
                        barmode=bar_mode, title=title,
                        color_discrete_sequence=palette_sequence,
                    )
                if as_percent:
                    bar_fig.update_yaxes(ticksuffix="%")
                st.plotly_chart(update_chart_design(bar_fig), use_container_width=True)
            else:
                st.info("No categorical columns available.")

        st.divider()

        # --- Correlation ---
        st.subheader("Correlation (numeric)")
        if len(numeric_cols) >= 2:
            corr = filtered_data[numeric_cols].corr(numeric_only=True)
            corr_fig = px.imshow(
                corr, text_auto=".2f", title="Correlation Matrix",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            )
            st.plotly_chart(update_chart_design(corr_fig, height=500), use_container_width=True)
            st.caption(sample_size_label(data, filtered_data, datetime_cols))
        else:
            st.info("Need at least 2 numeric columns for correlation.")

        st.divider()

        # --- Time Series ---
        st.subheader("Time Series")
        if datetime_cols and numeric_cols:
            t1, t2, t3, t4 = st.columns([0.3, 0.3, 0.2, 0.2])
            with t1:
                dt_col = st.selectbox("Date column", options=datetime_cols, index=0)
            with t2:
                val_col = st.selectbox("Value column", options=numeric_cols, index=0)
            with t3:
                freq = st.selectbox("Bucket", options=["D", "W", "M"], index=1)
            with t4:
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
                    ts_out, x=dt_col, y="value",
                    title=f"{agg.title()} of {val_col} over time ({freq})",
                    markers=True, line_shape="spline",
                    color_discrete_sequence=[palette_sequence[0]],
                )
                ts_fig.update_traces(line=dict(width=3))
                st.plotly_chart(update_chart_design(ts_fig), use_container_width=True)
                st.caption(sample_size_label(data, filtered_data, datetime_cols))
        else:
            st.info("Time series requires at least one datetime and one numeric column.")

        st.divider()

        # --- Scatter Explorer ---
        st.subheader("Scatter Explorer")
        if len(numeric_cols) >= 2:
            sc1, sc2, sc3 = st.columns([0.4, 0.4, 0.2])
            with sc1:
                x_col = st.selectbox("X-axis", options=numeric_cols, index=0, key="viz_scatter_x")
            with sc2:
                y_candidates = [c for c in numeric_cols if c != x_col] or numeric_cols
                y_col = st.selectbox("Y-axis", options=y_candidates, index=0, key="viz_scatter_y")
            with sc3:
                max_points = st.slider("Max points", 200, 3000, 1500, 100, key="viz_scatter_n")

            scatter_df = filtered_data[[x_col, y_col]].dropna().head(max_points)
            if scatter_df.empty:
                st.info("Not enough non-null values for scatter plot.")
            else:
                scatter_fig = px.scatter(
                    scatter_df, x=x_col, y=y_col,
                    title=f"{y_col} vs {x_col}",
                    color_discrete_sequence=palette_sequence,
                )
                scatter_fig.update_traces(marker=dict(size=8, opacity=0.6))
                st.plotly_chart(update_chart_design(scatter_fig), use_container_width=True)
                st.caption(f"Scatter sample: n = {len(scatter_df):,}")
        else:
            st.info("Need at least two numeric columns for scatter exploration.")

# ---------- COMPARE ----------
with tab_compare:
    st.subheader("Filtered vs Full Dataset")
    if filtered_data.empty:
        st.warning("No comparison available — filtered data is empty.")
    else:
        st.caption(sample_size_label(data, filtered_data, datetime_cols))

        compare_df = comparison_metrics_table(data, filtered_data, numeric_cols)
        if compare_df.empty:
            st.info("No numeric metrics available for comparison.")
        else:
            metric_options = compare_df["metric"].tolist()
            metric_choice = st.selectbox("Metric to compare", options=metric_options, index=0, key="cmp_metric_choice")
            chosen_row = compare_df[compare_df["metric"] == metric_choice].iloc[0]

            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Full-data mean", f"{chosen_row['full_mean']:.3g}")
            cm2.metric("Filtered mean", f"{chosen_row['filtered_mean']:.3g}")
            cm3.metric("Delta vs full", f"{chosen_row['delta']:.3g}", f"{chosen_row['delta_pct']:.1f}%")

            st.dataframe(compare_df, use_container_width=True, height=260)

            comp_fig = px.bar(
                compare_df.head(10),
                x="metric", y=["full_mean", "filtered_mean"],
                barmode="group",
                title="Filtered vs full means (top 10 metrics)",
                color_discrete_sequence=get_palette_sequence("Safe"),
            )
            st.plotly_chart(update_chart_design(comp_fig), use_container_width=True)

        st.divider()

        # --- Time Period Comparison ---
        st.subheader("Time Period Comparison")
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
                    .resample(period_freq).mean().rename("value").reset_index()
                )
                full_agg["dataset"] = "Full"

                filt_agg = (
                    filt_ts.set_index(period_dt_col)[period_metric_col]
                    .resample(period_freq).mean().rename("value").reset_index()
                )
                filt_agg["dataset"] = "Filtered"

                comp_time = pd.concat([full_agg, filt_agg], ignore_index=True)
                time_fig = px.line(
                    comp_time, x=period_dt_col, y="value", color="dataset",
                    title=f"Mean {period_metric_col} over time: filtered vs full",
                    color_discrete_sequence=get_palette_sequence("Bold"),
                    markers=True, line_shape="spline",
                )
                time_fig.update_traces(line=dict(width=3))
                st.plotly_chart(update_chart_design(time_fig), use_container_width=True)
        else:
            st.info("Need one datetime and one numeric column for time period comparison.")

        st.divider()

        # --- Benchmark ---
        st.subheader("Benchmark Target Tracking")
        if numeric_cols:
            b1, b2, b3 = st.columns([0.35, 0.35, 0.3])
            with b1:
                benchmark_col = st.selectbox("Benchmark metric", options=numeric_cols, key="cmp_bench_col")
            current_metric = (
                float(filtered_data[benchmark_col].dropna().mean())
                if filtered_data[benchmark_col].dropna().shape[0]
                else 0.0
            )
            with b2:
                benchmark_target = st.number_input("Target value", value=current_metric, key="cmp_bench_target")
            with b3:
                goal_direction = st.selectbox(
                    "Goal direction",
                    options=["Lower is better", "Higher is better"],
                    key="cmp_goal_direction",
                )

            if goal_direction == "Lower is better":
                met_target = current_metric <= benchmark_target
                gap = current_metric - benchmark_target
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

# ---------- FINDINGS ----------
with tab_findings:
    st.subheader("Automated Insights")
    insights = generate_findings(data, filtered_data, numeric_cols, categorical_cols, datetime_cols)
    for idx, insight in enumerate(insights, start=1):
        st.write(f"**{idx}.** {insight}")
    st.caption("Insights are heuristic and intended to accelerate exploratory analysis.")

# ---------- AI ANALYSIS ----------
with tab_ai:
    st.subheader("AI-Powered Analysis")
    gemini_key = get_gemini_key() if HAS_GEMINI else ""
    gemini_ready = bool(HAS_GEMINI and gemini_key)

    if not HAS_GEMINI:
        st.info("The `google-generativeai` package is not installed. AI chat and report features are unavailable, but anomaly detection and forecasting work below.")
    elif not gemini_key:
        st.info("Enter your Google Gemini API key in the sidebar to unlock AI chat and auto-reports. Anomaly detection and forecasting work without a key.")

    # Build context for Gemini (used by both chat and report)
    ai_context = build_gemini_context(
        filtered_data, numeric_cols, categorical_cols, datetime_cols, filter_summaries, kpis,
    )

    # ---- 1. CHAT WITH YOUR DATA ----
    st.markdown("### 💬 Chat With Your Data")
    if gemini_ready:
        st.caption(
            "Ask questions about your data in plain English. "
            "Only column names, summary stats, and 5 sample rows are sent to the AI — never the full dataset."
        )

        # Initialize chat history
        if "ai_chat_history" not in st.session_state:
            st.session_state["ai_chat_history"] = []

        # Display chat history
        for msg in st.session_state["ai_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_question = st.chat_input("Ask about your data... e.g. 'Which routes use the most energy?'")
        if user_question:
            st.session_state["ai_chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        model = init_gemini(gemini_key)
                        system_prompt = (
                            "You are a data analyst assistant embedded in a business analytics dashboard. "
                            "Answer questions about the user's dataset concisely and accurately. "
                            "Use the data context provided. Format responses with markdown. "
                            "If you cannot determine something from the data, say so.\n\n"
                            f"DATA CONTEXT:\n{ai_context}"
                        )
                        # Build conversation for multi-turn
                        messages = [{"role": "user", "parts": [system_prompt]}]
                        messages.append({"role": "model", "parts": ["Understood. I have your dataset context. Ask me anything about your data."]})
                        for past in st.session_state["ai_chat_history"][:-1]:
                            role = "user" if past["role"] == "user" else "model"
                            messages.append({"role": role, "parts": [past["content"]]})
                        messages.append({"role": "user", "parts": [user_question]})

                        chat = model.start_chat(history=messages[:-1])
                        response = chat.send_message(user_question)
                        answer = response.text
                    except Exception as e:
                        answer = f"Error communicating with Gemini: {e}"

                st.markdown(answer)
                st.session_state["ai_chat_history"].append({"role": "assistant", "content": answer})

        if st.session_state.get("ai_chat_history"):
            if st.button("Clear chat history", key="ai_clear_chat"):
                st.session_state["ai_chat_history"] = []
                st.rerun()
    else:
        st.caption("Configure your Gemini API key in the sidebar to enable data chat.")

    st.divider()

    # ---- 2. AUTO-GENERATED REPORT ----
    st.markdown("### 📝 AI-Generated Report")
    if gemini_ready:
        st.caption("One-click narrative analysis of your current dataset and filters.")

        if st.button("Generate AI Report", key="ai_gen_report", type="primary"):
            with st.spinner("Generating narrative report..."):
                try:
                    model = init_gemini(gemini_key)

                    # Include existing findings
                    findings_list = generate_findings(
                        data, filtered_data, numeric_cols, categorical_cols, datetime_cols
                    )
                    findings_text = "\n".join(f"- {f}" for f in findings_list)

                    report_prompt = (
                        "You are a senior data analyst. Write a professional narrative report about the following dataset. "
                        "Structure it with these sections:\n"
                        "1. **Executive Summary** (2-3 sentences)\n"
                        "2. **Data Overview** (schema, quality, completeness)\n"
                        "3. **Key Findings** (the most important patterns)\n"
                        "4. **Anomalies & Concerns** (data quality issues or unusual patterns)\n"
                        "5. **Recommendations** (actionable next steps)\n\n"
                        "Be specific with numbers. Use markdown formatting.\n\n"
                        f"DATA CONTEXT:\n{ai_context}\n\n"
                        f"AUTOMATED FINDINGS:\n{findings_text}"
                    )
                    response = model.generate_content(report_prompt)
                    report_text = response.text
                    st.session_state["ai_report"] = report_text
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if st.session_state.get("ai_report"):
            st.markdown(st.session_state["ai_report"])
            st.download_button(
                "Download AI Report (.md)",
                st.session_state["ai_report"],
                "ai_analysis_report.md",
                "text/markdown",
                use_container_width=True,
            )
    else:
        st.caption("Configure your Gemini API key in the sidebar to generate AI reports.")

    st.divider()

    # ---- 3. ANOMALY DETECTION ----
    st.markdown("### 🔍 Anomaly Detection")
    if HAS_SKLEARN:
        st.caption("Uses Isolation Forest to flag unusual data points. No API key required.")

        if numeric_cols and not filtered_data.empty:
            anom_cols = st.multiselect(
                "Select numeric columns to scan",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="ai_anom_cols",
            )
            anom_sensitivity = st.slider(
                "Sensitivity (expected anomaly %)",
                min_value=1, max_value=20, value=5, step=1,
                key="ai_anom_sensitivity",
                help="Higher = more points flagged as anomalies",
            )

            if anom_cols and st.button("Run Anomaly Detection", key="ai_run_anom"):
                with st.spinner("Scanning for anomalies..."):
                    anom_df, anom_count = run_anomaly_detection(
                        filtered_data, anom_cols, contamination=anom_sensitivity / 100.0
                    )
                    st.session_state["anom_result"] = anom_df
                    st.session_state["anom_count"] = anom_count
                    st.session_state["anom_cols_used"] = anom_cols

            if st.session_state.get("anom_result") is not None and st.session_state.get("anom_count", 0) > 0:
                anom_df = st.session_state["anom_result"]
                anom_count = st.session_state["anom_count"]
                anom_cols_used = st.session_state.get("anom_cols_used", [])
                total = len(anom_df)
                anom_pct = (anom_count / total * 100) if total else 0

                am1, am2, am3 = st.columns(3)
                am1.metric("Anomalies Detected", f"{anom_count:,}")
                am2.metric("Anomaly Rate", f"{anom_pct:.1f}%")
                am3.metric("Normal Points", f"{total - anom_count:,}")

                # Scatter plot if 2+ columns
                if len(anom_cols_used) >= 2:
                    scatter_anom = anom_df.copy()
                    scatter_anom["Status"] = scatter_anom["_anomaly"].map({True: "Anomaly", False: "Normal"})
                    anom_fig = px.scatter(
                        scatter_anom.head(2000),
                        x=anom_cols_used[0], y=anom_cols_used[1],
                        color="Status",
                        color_discrete_map={"Normal": "#6366f1", "Anomaly": "#ef4444"},
                        title=f"Anomalies: {anom_cols_used[0]} vs {anom_cols_used[1]}",
                        opacity=0.7,
                    )
                    anom_fig.update_traces(
                        marker=dict(size=8),
                        selector=dict(name="Anomaly"),
                    )
                    st.plotly_chart(update_chart_design(anom_fig), use_container_width=True)

                # Show anomaly rows
                with st.expander(f"View {anom_count} anomalous rows", expanded=False):
                    anomalous_rows = anom_df[anom_df["_anomaly"]].drop(columns=["_anomaly"])
                    st.dataframe(anomalous_rows, use_container_width=True, height=300)

                    csv_anom = anomalous_rows.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download anomalies CSV", csv_anom,
                        "anomalies.csv", "text/csv", use_container_width=True,
                    )
            elif st.session_state.get("anom_count") == 0 and st.session_state.get("anom_result") is not None:
                st.success("No anomalies detected at the current sensitivity level.")
        else:
            st.info("Need numeric columns and data to run anomaly detection.")
    else:
        st.info("Install `scikit-learn` to enable anomaly detection.")

    st.divider()

    # ---- 4. TREND FORECASTING ----
    st.markdown("### 📈 Trend Forecasting")
    st.caption("Fits a trend line to past data and extends it forward. Best for smooth trends; weak for noisy metrics. No API key required.")

    if datetime_cols and numeric_cols and not filtered_data.empty:
        fc1, fc2, fc3, fc4 = st.columns([0.3, 0.3, 0.2, 0.2])
        with fc1:
            fc_dt_col = st.selectbox("Date column", options=datetime_cols, index=0, key="ai_fc_dt")
        with fc2:
            fc_val_col = st.selectbox("Metric to forecast", options=numeric_cols, index=0, key="ai_fc_val")
        with fc3:
            fc_periods = st.slider("Forecast periods", 4, 52, 12, key="ai_fc_periods")
        with fc4:
            fc_degree = st.selectbox("Trend type", options=[1, 2, 3], format_func=lambda d: {1: "Linear", 2: "Quadratic", 3: "Cubic"}[d], key="ai_fc_degree")

        if st.button("Run Forecast", key="ai_run_fc"):
            with st.spinner("Forecasting..."):
                fc_result, r_sq = forecast_trend(
                    filtered_data, fc_dt_col, fc_val_col,
                    periods=fc_periods, degree=fc_degree,
                )
                if fc_result.empty:
                    st.warning("Not enough data points to forecast (need at least 5).")
                else:
                    st.session_state["fc_result"] = fc_result
                    st.session_state["fc_r_squared"] = r_sq
                    st.session_state["fc_dt_col"] = fc_dt_col
                    st.session_state["fc_val_col"] = fc_val_col

        if st.session_state.get("fc_result") is not None and not st.session_state["fc_result"].empty:
            fc_result = st.session_state["fc_result"]
            r_sq = st.session_state["fc_r_squared"]
            fc_dt = st.session_state["fc_dt_col"]
            fc_val = st.session_state["fc_val_col"]

            fm1, fm2, fm3 = st.columns(3)
            fm1.metric("R-squared", f"{r_sq:.3f}")
            trend_dir = "Upward" if fc_result["fitted"].iloc[-1] > fc_result["fitted"].iloc[0] else "Downward"
            fm2.metric("Trend Direction", trend_dir)
            fm3.metric("Forecast Periods", f"{fc_periods}")

            # Build the chart
            fc_fig = go.Figure()

            # Historical actual values
            hist = fc_result[fc_result["type"] == "Historical"]
            fc_fig.add_trace(go.Scatter(
                x=hist[fc_dt], y=hist["actual"],
                mode="markers", name="Actual",
                marker=dict(color="#6366f1", size=6, opacity=0.6),
            ))

            # Fitted line (historical)
            fc_fig.add_trace(go.Scatter(
                x=hist[fc_dt], y=hist["fitted"],
                mode="lines", name="Fitted",
                line=dict(color="#10b981", width=2),
            ))

            # Forecast line
            fcast = fc_result[fc_result["type"] == "Forecast"]
            fc_fig.add_trace(go.Scatter(
                x=fcast[fc_dt], y=fcast["fitted"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#f59e0b", width=3, dash="dash"),
                marker=dict(size=6),
            ))

            # Confidence band (forecast only)
            fc_fig.add_trace(go.Scatter(
                x=pd.concat([fcast[fc_dt], fcast[fc_dt][::-1]]),
                y=pd.concat([fcast["upper"], fcast["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence",
                showlegend=True,
            ))

            fc_fig.update_layout(title=f"Forecast: {fc_val}")
            st.plotly_chart(update_chart_design(fc_fig, height=450), use_container_width=True)

            # Plain English chart narration
            st.info(
                "**How to read this chart:** Dots are real data values. "
                "The solid green line is the trend through the past. "
                "The dashed orange line is the app's best guess going forward, "
                "and the shaded band shows the range of likely values (wider = less certain)."
            )

            if r_sq < 0.2:
                st.warning(
                    f"⚠️ **Low R² ({r_sq:.3f}) — this metric is very noisy over time.** "
                    "The forecast may not be useful. Try weekly/monthly averages or a different metric."
                )
            elif r_sq < 0.5:
                st.caption("Moderate fit. Forecast is directionally useful but treat exact values with caution.")
            else:
                st.caption("Strong fit. Forecast confidence is relatively high for near-term projections.")
    else:
        st.info("Trend forecasting requires at least one datetime and one numeric column.")

# ---------- EXPORT ----------
with tab_export:
    st.subheader("Export Results")
    if filtered_data.empty:
        st.warning("Filtered dataset is empty — nothing to export.")
    else:
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_bytes = filtered_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered CSV", csv_bytes,
                "filtered_data.csv", "text/csv", use_container_width=True,
            )
        with col_ex2:
            source_info = {
                "source_system": st.session_state.get("src_system", ""),
                "export_date": st.session_state.get("src_export_date", ""),
                "owner": st.session_state.get("src_owner", ""),
                "source_url": st.session_state.get("src_url", ""),
                "limitations": st.session_state.get("src_limitations", ""),
            }
            insights = generate_findings(data, filtered_data, numeric_cols, categorical_cols, datetime_cols)
            summary = build_summary_payload(
                data, filtered_data, filter_summaries, insights, meta, source_info, datetime_cols,
            )
            summary_json = json.dumps(summary, indent=2)
            st.download_button(
                "Download summary (JSON)", summary_json,
                "analysis_summary.json", "application/json", use_container_width=True,
            )

        col_ex3, col_ex4 = st.columns(2)
        with col_ex3:
            summary_html = build_html_report(summary)
            st.download_button(
                "Download report (HTML)", summary_html,
                "analysis_summary.html", "text/html", use_container_width=True,
            )
        with col_ex4:
            st.download_button(
                "Download README", build_readme_text(),
                "README.md", "text/markdown", use_container_width=True,
            )

        st.caption(
            f"Export includes {len(filtered_data):,} rows, {filtered_data.shape[1]} columns, "
            f"and a narrative summary."
        )

    # --- Save / Load Session ---
    st.markdown("---")
    st.subheader("💾 Save / Load Session")
    st.caption(
        "Save your current session (cleaning history, business context, AI chat) to a JSON file. "
        "Restore it later by uploading the file alongside your CSV."
    )

    # Collect session state to persist
    _SESSION_KEYS = [
        "cleaning_history", "src_system", "src_export_date",
        "src_owner", "src_url", "src_limitations", "ai_chat_history",
    ]
    session_payload: dict = {}
    for _sk in _SESSION_KEYS:
        val = st.session_state.get(_sk)
        if val is not None:
            session_payload[_sk] = val
    session_payload["_saved_at"] = datetime.now().isoformat()
    session_payload["_app_version"] = "analytics-dashboard-v2"

    session_json_str = json.dumps(session_payload, indent=2, default=str)
    st.download_button(
        "Save Session",
        data=session_json_str,
        file_name="dashboard_session.json",
        mime="application/json",
        use_container_width=True,
        key="export_save_session",
    )

# ---------- HELP ----------
with tab_help:
    st.subheader("How to Use This Dashboard")
    st.markdown(
        "1. **Upload CSV(s)** — Upload one or more CSV files. Multiple files can be stacked (same columns) or merged (shared key column)\n"
        "2. **Restore session** — Optionally upload a `dashboard_session.json` to restore a previous session's context and history\n"
        "3. **Clean data** — Optionally clean data from the sidebar (remove outliers, fill missing, etc.)\n"
        "4. **Apply filters** — Filter categorical, numeric, and date columns\n"
        "5. **Review insights** — Explore KPIs, visualizations, findings, and comparisons\n"
        "6. **Export & Save** — Download filtered data, summaries, and save your session for later"
    )

    st.markdown("**Keyboard-friendly tips**")
    st.write("• Press `/` then type widget labels to jump quickly in browser search")
    st.write("• Use the clear-filter buttons when no rows match")
    st.write("• Keep filters narrow for focused insights, broad for trend analysis")

    st.markdown("**Expected input**")
    st.write("• CSV with headers")
    st.write("• Mixed data types are supported")
    st.write("• Datetime columns are auto-detected when parse confidence is high")

    st.markdown("**Troubleshooting**")
    st.write("• Upload errors: verify file is CSV and below 200MB")
    st.write("• Empty charts: verify required numeric/categorical columns exist")
    st.write("• Unexpected results: review cleaning history and active filters")

    with st.expander("View README", expanded=False):
        st.code(build_readme_text(), language="markdown")
