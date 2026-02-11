# Document Intelligence Dashboard

Interactive Streamlit app that turns a source document (DOCX/PDF/TXT) into a clickable intelligence workspace with overview, outline navigation, entities/relationships, requirements rubric, visual insights, and grounded Q&A.

## Features

- 6 required tabs:
  - Overview
  - Outline Navigator
  - Entities & Relationships
  - Requirements / Rubric Builder
  - Visual Insights
  - Q&A Workbench
- Global command palette trigger: `Ctrl+K` / `Cmd+K`
- Click-through behavior from entities, requirements, and charts to source sections
- Grounded Q&A with citations and refusal behavior for unsupported claims (`Not found in document`)
- Export buttons for entities, requirements, and analysis summary
- Runtime parsing cache and schema validation (Pydantic)
- Graceful fallback for sparse/malformed structure

## Project Structure

- `app.py` - Streamlit app UI
- `src/models.py` - typed schema models
- `src/parsers.py` - DOCX/PDF/TXT parsing and normalization
- `src/sectionizer.py` - section detection heuristics
- `src/extractors.py` - entities/requirements/relationships/insights
- `src/qa.py` - grounded Q&A retrieval and refusal logic
- `src/pipeline.py` - `analyze_document(...)` API
- `src/exporters.py` - CSV/JSON/Markdown exports
- `scripts/self_check.py` - acceptance-aligned pipeline checks
- `tests/test_pipeline_validation.py` - lightweight tests
- `run.sh` - one-command local launcher

## Install

```bash
cd /Users/edwinbrown/document-intelligence-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (One Command)

```bash
cd /Users/edwinbrown/document-intelligence-dashboard
./run.sh --doc "/Users/edwinbrown/Downloads/STA 9708 LN3.1 Rules of Probability 2-10-2026.docx"
```

The script will:
- create/reuse `.venv`
- install dependencies
- bind to `127.0.0.1`
- start at port `3001` (auto-increments if busy)

## Where To Place Document

Default path loaded automatically:

`/Users/edwinbrown/Downloads/STA 9708 LN3.1 Rules of Probability 2-10-2026.docx`

You can override via:
- `./run.sh --doc /path/to/your/file`
- or use in-app file upload.

## Validation / Self-Check

Run acceptance-aligned checks:

```bash
cd /Users/edwinbrown/document-intelligence-dashboard
source .venv/bin/activate
python scripts/self_check.py --doc "/Users/edwinbrown/Downloads/STA 9708 LN3.1 Rules of Probability 2-10-2026.docx"
pytest -q
```

## Deploy

### GitHub

```bash
cd /Users/edwinbrown/document-intelligence-dashboard
git init
git add .
git commit -m "Initial Document Intelligence Dashboard"
gh repo create navybrown1/document-intelligence-dashboard --public --source=. --remote=origin --push
```

### Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Create a **new app** from `navybrown1/document-intelligence-dashboard`
3. Branch: `main`
4. Main file: `app.py`
5. Deploy

After first setup, pushes to `main` auto-redeploy.

## Screenshot Instructions (Optional)

- Start app with `./run.sh`
- Open browser at shown local URL
- Capture each tab for documentation:
  - Overview
  - Outline Navigator
  - Entities & Relationships
  - Requirements / Rubric Builder
  - Visual Insights
  - Q&A Workbench

## Design Notes

### Entity Extraction

- Hybrid approach:
  - Regex (dates, probability notation, fractions, set expressions)
  - Lexical concept dictionary (mutually exclusive, union, intersection, etc.)
  - Contextual cues (Prof/College, modal requirement signals)
- Every entity stores citations with section + block snippet for jump navigation.

### Requirements Detection

- Sentence-level detection using modal/rule cues:
  - `must`, `should`, `may`, `required`, `rule`, `defined as`, conditionals (`if...then`)
- Priority mapping:
  - `must/required` -> High
  - `should` -> Medium
  - `may` -> Low
  - definitions/rules default -> Medium
- Each requirement includes:
  - rationale
  - verification method
  - citation link

### Confidence and Risk Heuristics

- Confidence flags:
  - `explicit`: directly quoted or normative statement in source text
  - `inferred`: synthesis from frequency or heuristic scoring
- Risk/ambiguity scores by section:
  - hedges (`may`, `might`, `could`)
  - uncertainty markers (`not`, `?`, etc.)
  - contrast terms (`however`, `but`, etc.)
  - parse-confidence penalty when heading structure is weak

