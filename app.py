from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import fuzz, process
try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:  # pragma: no cover - optional dependency behavior
    streamlit_js_eval = None

try:
    from streamlit_plotly_events import plotly_events
except Exception:  # pragma: no cover - optional dependency behavior
    plotly_events = None

try:
    from streamlit_tree_select import tree_select
except Exception:  # pragma: no cover - optional dependency behavior
    tree_select = None

from src.exporters import (
    entities_to_dataframe,
    export_entities_csv,
    export_entities_json,
    export_requirements_csv,
    export_requirements_json,
    export_summary_json,
    export_summary_markdown,
    requirements_to_dataframe,
)
from src.models import AnalysisResult, Citation, Entity, RequirementItem, Section
from src.pipeline import analyze_document
from src.qa import answer_question


CACHE_VERSION = "2026-02-11-1"
DEFAULT_DOC_PATH = os.environ.get(
    "DOCUMENT_PATH",
    "/Users/edwinbrown/Downloads/STA 9708 LN3.1 Rules of Probability 2-10-2026.docx",
)


def init_state() -> None:
    defaults = {
        "selected_section_id": "",
        "selected_entity_id": "",
        "selected_requirement_id": "",
        "outline_keyword": "",
        "outline_section_types": [],
        "outline_entity_types": [],
        "command_palette_open": False,
        "command_notice": "",
        "qa_history": [],
        "palette_query": "",
        "last_cmdk_ts": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def highlight_text(text: str, terms: List[str]) -> str:
    html = text
    for term in terms:
        term = term.strip()
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        html = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", html)
    return html


@st.cache_data(show_spinner=False)
def load_analysis_from_bytes(file_bytes: bytes, filename: str, cache_version: str) -> AnalysisResult:
    return analyze_document(file_bytes=file_bytes, filename=filename)


@st.cache_data(show_spinner=False)
def load_analysis_from_path(path: str, cache_version: str, mtime: float) -> AnalysisResult:
    _ = mtime
    return analyze_document(file_path=path)


def get_analysis(uploaded_file) -> Tuple[Optional[AnalysisResult], Optional[str]]:
    if uploaded_file is not None:
        try:
            data = uploaded_file.getvalue()
            analysis = load_analysis_from_bytes(data, uploaded_file.name, CACHE_VERSION)
            return analysis, uploaded_file.name
        except Exception as exc:
            st.error(f"Failed to parse uploaded file: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)
            return None, None

    default = Path(DEFAULT_DOC_PATH)
    if default.exists():
        try:
            analysis = load_analysis_from_path(str(default), CACHE_VERSION, default.stat().st_mtime)
            return analysis, default.name
        except Exception as exc:
            st.error(f"Failed to parse default document: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)
            return None, None

    return None, None


def section_maps(analysis: AnalysisResult) -> Tuple[Dict[str, Section], Dict[str, object], Dict[str, List[Entity]]]:
    section_by_id = {section.id: section for section in analysis.sections}
    block_by_id = {block.id: block for block in analysis.blocks}
    entities_by_section: Dict[str, List[Entity]] = {section.id: [] for section in analysis.sections}
    for entity in analysis.entities:
        seen = set()
        for citation in entity.citations:
            if citation.section_id in entities_by_section and citation.section_id not in seen:
                entities_by_section[citation.section_id].append(entity)
                seen.add(citation.section_id)
    return section_by_id, block_by_id, entities_by_section


def ensure_selected_section(analysis: AnalysisResult) -> str:
    if not analysis.sections:
        return ""
    section_ids = {section.id for section in analysis.sections}
    selected = st.session_state.get("selected_section_id", "")
    if selected in section_ids:
        return selected
    fallback = analysis.sections[0].id
    st.session_state["selected_section_id"] = fallback
    return fallback


def jump_to_citation(citation: Citation) -> None:
    st.session_state["selected_section_id"] = citation.section_id


def execute_command(command_value: str, analysis: AnalysisResult) -> None:
    if command_value.startswith("section:"):
        st.session_state["selected_section_id"] = command_value.split(":", 1)[1]
        st.session_state["command_notice"] = "Section jump queued. Open Outline Navigator to view it."
        return

    if command_value.startswith("entity:"):
        entity_id = command_value.split(":", 1)[1]
        st.session_state["selected_entity_id"] = entity_id
        entity = next((item for item in analysis.entities if item.id == entity_id), None)
        if entity and entity.citations:
            st.session_state["selected_section_id"] = entity.citations[0].section_id
        st.session_state["command_notice"] = "Entity jump queued."
        return

    if command_value == "action:clear_filters":
        st.session_state["outline_keyword"] = ""
        st.session_state["outline_section_types"] = []
        st.session_state["outline_entity_types"] = []
        st.session_state["command_notice"] = "Filters cleared."
        return

    if command_value == "action:reset_jumps":
        st.session_state["selected_entity_id"] = ""
        st.session_state["selected_requirement_id"] = ""
        st.session_state["command_notice"] = "Selection state reset."
        return


def build_command_options(analysis: AnalysisResult) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for section in analysis.sections:
        options.append((f"Go to section: {section.title}", f"section:{section.id}"))
    for entity in analysis.entities[:80]:
        options.append((f"Go to entity: {entity.label} ({entity.type})", f"entity:{entity.id}"))
    options.extend(
        [
            ("Action: Clear search filters", "action:clear_filters"),
            ("Action: Reset jump selections", "action:reset_jumps"),
        ]
    )
    return options


def command_palette_ui(analysis: AnalysisResult) -> None:
    components.html(
        """
        <script>
        if (!window.parent.__docintelCmdkInstalled) {
            window.parent.__docintelCmdkInstalled = true;
            window.parent.document.addEventListener('keydown', function(event) {
                const key = (event.key || '').toLowerCase();
                if ((event.ctrlKey || event.metaKey) && key === 'k') {
                    event.preventDefault();
                    window.parent.localStorage.setItem('docintel_cmdk_ts', String(Date.now()));
                }
            });
        }
        </script>
        """,
        height=0,
        width=0,
    )

    shortcut_fired = False
    if streamlit_js_eval is not None:
        event_stamp = streamlit_js_eval(
            js_expressions="""
            (function() {
                const key = 'docintel_cmdk_ts';
                const value = window.localStorage.getItem(key) || '';
                if (value) { window.localStorage.removeItem(key); }
                return value;
            })()
            """,
            key="command_palette_shortcut_eval",
        )
        if event_stamp and str(event_stamp) != st.session_state.get("last_cmdk_ts", ""):
            st.session_state["last_cmdk_ts"] = str(event_stamp)
            shortcut_fired = True

    opened = st.button(
        "Open Command Palette (Ctrl/Cmd+K)",
        key="command_palette_button",
        type="secondary",
        use_container_width=True,
    )
    if opened or shortcut_fired:
        st.session_state["command_palette_open"] = True

    if not st.session_state.get("command_palette_open"):
        return

    with st.container(border=True):
        st.markdown("### Command Palette")
        query = st.text_input("Search commands", key="palette_query")
        command_options = build_command_options(analysis)
        labels = [label for label, _ in command_options]

        if query.strip():
            results = process.extract(query, labels, scorer=fuzz.WRatio, limit=12)
            labels = [label for label, _, _ in results]

        if not labels:
            st.info("No matching commands.")
        else:
            selected_label = st.selectbox("Command", labels, key="palette_selected")
            command_value = next(value for label, value in command_options if label == selected_label)
            col_run, col_close = st.columns([1, 1])
            with col_run:
                if st.button("Run Command", use_container_width=True):
                    execute_command(command_value, analysis)
            with col_close:
                if st.button("Close Palette", use_container_width=True):
                    st.session_state["command_palette_open"] = False
                    st.session_state["palette_query"] = ""


def render_overview_tab(analysis: AnalysisResult) -> None:
    st.subheader("Executive Summary")
    st.write(analysis.summary or "No summary generated.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key Takeaways")
        if analysis.takeaways:
            for idx, takeaway in enumerate(analysis.takeaways[:8], start=1):
                st.markdown(
                    f"{idx}. {takeaway.text}  \n"
                    f"`{takeaway.confidence}` | Source: `{takeaway.citation.section_title}`"
                )
                if st.button(
                    f"Jump to source ({idx})",
                    key=f"takeaway_jump_{idx}",
                ):
                    jump_to_citation(takeaway.citation)
        else:
            st.info("No takeaways extracted.")

    with c2:
        st.markdown("#### Confidence Flags")
        if analysis.confidence_flags:
            for idx, flag in enumerate(analysis.confidence_flags[:8], start=1):
                st.caption(
                    f"[{flag.confidence}] {flag.statement}\nReason: {flag.reason}\nSource: {flag.citation.section_title}"
                )
        else:
            st.info("No confidence flags available.")

    st.markdown("#### Summary Exports")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export Analysis Summary (JSON)",
            data=export_summary_json(analysis),
            file_name="analysis_summary.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Export Analysis Summary (Markdown)",
            data=export_summary_markdown(analysis),
            file_name="analysis_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_outline_tab(analysis: AnalysisResult, section_by_id: Dict[str, Section], block_by_id: Dict[str, object], entities_by_section: Dict[str, List[Entity]]) -> None:
    all_section_types = sorted({section.section_type for section in analysis.sections})
    all_entity_types = sorted({entity.type for entity in analysis.entities})

    col_filters = st.columns([2, 2, 2])
    with col_filters[0]:
        st.session_state["outline_keyword"] = st.text_input(
            "Keyword",
            value=st.session_state.get("outline_keyword", ""),
            key="outline_keyword_input",
        )
    with col_filters[1]:
        st.session_state["outline_section_types"] = st.multiselect(
            "Section Type",
            all_section_types,
            default=st.session_state.get("outline_section_types", []),
            key="outline_section_types_input",
        )
    with col_filters[2]:
        st.session_state["outline_entity_types"] = st.multiselect(
            "Entity Type",
            all_entity_types,
            default=st.session_state.get("outline_entity_types", []),
            key="outline_entity_types_input",
        )

    keyword = st.session_state.get("outline_keyword", "").strip().lower()
    selected_section_types = set(st.session_state.get("outline_section_types", []))
    selected_entity_types = set(st.session_state.get("outline_entity_types", []))

    filtered_sections: List[Section] = []
    for section in analysis.sections:
        if selected_section_types and section.section_type not in selected_section_types:
            continue
        if keyword and keyword not in section.title.lower() and keyword not in section.text.lower():
            continue
        if selected_entity_types:
            section_entities = entities_by_section.get(section.id, [])
            if not any(entity.type in selected_entity_types for entity in section_entities):
                continue
        filtered_sections.append(section)

    if not filtered_sections:
        st.warning("No sections matched your filters.")
        return

    left, right = st.columns([1, 2])

    selected_section_id = ensure_selected_section(analysis)

    with left:
        st.markdown("#### Outline Tree")
        nodes = [
            {
                "label": section.title,
                "value": section.id,
                "children": [],
                "expanded": True,
            }
            for section in filtered_sections
        ]
        if tree_select is not None:
            tree = tree_select(
                nodes,
                checked=[selected_section_id] if selected_section_id in [s.id for s in filtered_sections] else [],
                expanded=[node["value"] for node in nodes],
                no_cascade=True,
                key="outline_tree",
            )
            checked = tree.get("checked", []) if isinstance(tree, dict) else []
            if checked:
                st.session_state["selected_section_id"] = checked[0]
                selected_section_id = checked[0]
        else:
            section_titles = [section.title for section in filtered_sections]
            id_by_title = {section.title: section.id for section in filtered_sections}
            current_title = next(
                (section.title for section in filtered_sections if section.id == selected_section_id),
                section_titles[0],
            )
            title = st.selectbox("Section", section_titles, index=section_titles.index(current_title))
            st.session_state["selected_section_id"] = id_by_title[title]
            selected_section_id = id_by_title[title]

    with right:
        section = section_by_id.get(selected_section_id)
        if not section:
            section = filtered_sections[0]
            st.session_state["selected_section_id"] = section.id

        st.markdown(f"#### {section.title}")
        selected_entity = next((e for e in analysis.entities if e.id == st.session_state.get("selected_entity_id")), None)
        terms = []
        if keyword:
            terms.append(keyword)
        if selected_entity:
            terms.append(selected_entity.label)

        for block_id in section.block_ids:
            block = block_by_id.get(block_id)
            if not block:
                continue
            rendered = highlight_text(block.text, terms)
            st.markdown(f"<div><strong>[{block.index}]</strong> {rendered}</div>", unsafe_allow_html=True)

        st.caption(
            f"Section confidence: {section.parse_confidence:.2f} | "
            f"Risk: {section.risk_score:.2f} | Ambiguity: {section.ambiguity_score:.2f}"
        )


def build_relationship_figure(analysis: AnalysisResult) -> Tuple[go.Figure, List[str]]:
    graph = nx.Graph()
    node_labels: Dict[str, str] = {}

    for section in analysis.sections:
        graph.add_node(section.id, node_type="section")
        node_labels[section.id] = section.title

    for entity in analysis.entities[:80]:
        graph.add_node(entity.id, node_type="entity")
        node_labels[entity.id] = f"{entity.label} ({entity.type})"

    for edge in analysis.relationships:
        if edge.relation != "section_contains_entity":
            continue
        if edge.source in graph.nodes and edge.target in graph.nodes:
            graph.add_edge(edge.source, edge.target, weight=edge.weight)

    if graph.number_of_nodes() == 0:
        return go.Figure(), []

    pos = nx.spring_layout(graph, seed=42)
    edge_x: List[float] = []
    edge_y: List[float] = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 0.6, "color": "#9ca3af"},
        hoverinfo="none",
        mode="lines",
    )

    node_x: List[float] = []
    node_y: List[float] = []
    hover_text: List[str] = []
    node_color: List[str] = []
    node_size: List[int] = []
    node_order: List[str] = []

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_order.append(node)
        if data.get("node_type") == "section":
            node_color.append("#2563eb")
            node_size.append(19)
        else:
            node_color.append("#d97706")
            node_size.append(13)
        hover_text.append(node_labels[node])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker={"color": node_color, "size": node_size, "line": {"width": 1, "color": "#111827"}},
        hovertext=hover_text,
        hoverinfo="text",
        customdata=node_order,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin={"b": 0, "l": 0, "r": 0, "t": 0},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=520,
        ),
    )

    return fig, node_order


def render_entities_tab(analysis: AnalysisResult) -> None:
    st.markdown("#### Entity Explorer")
    entity_df = entities_to_dataframe(analysis)
    if entity_df.empty:
        st.info("No entities detected for this document.")
        return

    col_table, col_detail = st.columns([1.3, 1])

    selected_entity: Optional[Entity] = None

    with col_table:
        st.dataframe(entity_df, use_container_width=True, hide_index=True)
        entity_options = [f"{entity.label} ({entity.type})" for entity in analysis.entities]
        default_idx = 0
        if st.session_state.get("selected_entity_id"):
            for idx, entity in enumerate(analysis.entities):
                if entity.id == st.session_state["selected_entity_id"]:
                    default_idx = idx
                    break
        selected_label = st.selectbox("Inspect entity", entity_options, index=default_idx)
        selected_entity = analysis.entities[entity_options.index(selected_label)]
        st.session_state["selected_entity_id"] = selected_entity.id

        cexp1, cexp2 = st.columns(2)
        with cexp1:
            st.download_button(
                "Export Entities CSV",
                data=export_entities_csv(analysis),
                file_name="entities.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with cexp2:
            st.download_button(
                "Export Entities JSON",
                data=export_entities_json(analysis),
                file_name="entities.json",
                mime="application/json",
                use_container_width=True,
            )

    with col_detail:
        if selected_entity:
            st.markdown(f"**Label:** {selected_entity.label}")
            st.markdown(f"**Type:** `{selected_entity.type}`")
            st.markdown(f"**Mentions:** {selected_entity.mentions_count}")
            st.markdown(f"**First occurrence:** {selected_entity.first_occurrence}")
            st.markdown(f"**Last occurrence:** {selected_entity.last_occurrence}")
            st.markdown("**Source sections:**")
            for section in selected_entity.source_sections:
                st.caption(f"- {section}")
            for idx, citation in enumerate(selected_entity.citations[:4], start=1):
                st.caption(f"[{idx}] {citation.section_title}: {citation.snippet}")
                if st.button(f"Jump to citation {idx}", key=f"entity_jump_{selected_entity.id}_{idx}"):
                    jump_to_citation(citation)

    st.markdown("#### Relationship Graph")
    graph_fig, node_order = build_relationship_figure(analysis)
    if graph_fig.data:
        if plotly_events:
            selected = plotly_events(
                graph_fig,
                click_event=True,
                hover_event=True,
                key="relationship_graph",
                override_height=520,
            )
            if selected:
                point_index = selected[0].get("pointIndex")
                if point_index is not None and 0 <= point_index < len(node_order):
                    node_id = node_order[point_index]
                    if node_id.startswith("s"):
                        st.session_state["selected_section_id"] = node_id
                    elif node_id.startswith("ent_"):
                        st.session_state["selected_entity_id"] = node_id
                        entity = next((item for item in analysis.entities if item.id == node_id), None)
                        if entity and entity.citations:
                            st.session_state["selected_section_id"] = entity.citations[0].section_id
        else:
            st.plotly_chart(graph_fig, use_container_width=True)
            st.caption("Install streamlit-plotly-events for clickable graph points.")
    else:
        st.info("Relationship graph is empty for this document.")


def render_requirements_tab(analysis: AnalysisResult) -> None:
    st.markdown("#### Requirements Checklist")
    req_df = requirements_to_dataframe(analysis)
    if req_df.empty:
        st.info("No requirement/rule statements detected.")
    else:
        priority_filter = st.multiselect(
            "Priority",
            sorted(req_df["priority"].unique()),
            default=sorted(req_df["priority"].unique()),
        )
        filtered = req_df[req_df["priority"].isin(priority_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        for _, row in filtered.iterrows():
            with st.expander(f"{row['id']} | {row['priority']} | {row['section_title']}"):
                st.write(row["statement"])
                st.caption(f"Rationale: {row['rationale']}")
                st.caption(f"Verification: {row['verification_method']}")
                if st.button(f"Jump to source ({row['id']})", key=f"req_jump_{row['id']}"):
                    st.session_state["selected_section_id"] = row["section_id"]

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Export Checklist CSV",
            data=export_requirements_csv(analysis),
            file_name="requirements_checklist.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Export Checklist JSON",
            data=export_requirements_json(analysis),
            file_name="requirements_checklist.json",
            mime="application/json",
            use_container_width=True,
        )


def render_visual_insights_tab(analysis: AnalysisResult) -> None:
    section_by_id = {section.id: section for section in analysis.sections}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Timeline")
        if analysis.timeline:
            timeline_df = pd.DataFrame(analysis.timeline)
            timeline_df["order"] = range(1, len(timeline_df) + 1)
            fig = px.scatter(
                timeline_df,
                x="date",
                y="order",
                hover_data=["section_title", "snippet"],
                title="Dates and Deadlines Found in Document",
            )
            fig.update_traces(customdata=timeline_df[["section_id"]].values)
            if plotly_events:
                clicked = plotly_events(fig, click_event=True, key="timeline_click")
                if clicked:
                    point = clicked[0]
                    idx = point.get("pointIndex")
                    if idx is not None:
                        section_id = timeline_df.iloc[int(idx)]["section_id"]
                        if section_id:
                            st.session_state["selected_section_id"] = section_id
            else:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No parseable timeline dates were found.")

    with col2:
        st.markdown("#### Topic Frequency")
        topic_df = pd.DataFrame(
            [{"topic": key, "count": value} for key, value in analysis.topic_frequencies.items()]
        )
        if topic_df.empty:
            st.info("No topic frequency data available.")
        else:
            top_df = topic_df.head(20)
            fig = px.bar(top_df, x="topic", y="count", title="Most Frequent Concepts")
            section_map = {}
            for entity in analysis.entities:
                section_map[entity.label] = entity.citations[0].section_id if entity.citations else ""
            top_df = top_df.copy()
            top_df["section_id"] = top_df["topic"].map(section_map).fillna("")
            fig.update_traces(customdata=top_df[["section_id"]].values)
            fig.update_layout(xaxis_tickangle=-35)
            if plotly_events:
                clicked = plotly_events(fig, click_event=True, key="topic_click")
                if clicked:
                    idx = clicked[0].get("pointIndex")
                    if idx is not None:
                        section_id = top_df.iloc[int(idx)]["section_id"]
                        if section_id:
                            st.session_state["selected_section_id"] = section_id
            else:
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Risk / Ambiguity Heatmap")
    if analysis.sections:
        heat_df = pd.DataFrame(
            {
                "section": [section.title for section in analysis.sections],
                "risk": [section.risk_score for section in analysis.sections],
                "ambiguity": [section.ambiguity_score for section in analysis.sections],
                "section_id": [section.id for section in analysis.sections],
                "snippet": [section.text[:120] for section in analysis.sections],
            }
        )

        z = heat_df[["risk", "ambiguity"]].values
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["risk", "ambiguity"],
                y=heat_df["section"].tolist(),
                colorscale="YlOrRd",
                text=[
                    [f"{row['section']}: {row['snippet']}", f"{row['section']}: {row['snippet']}"]
                    for _, row in heat_df.iterrows()
                ],
                hovertemplate="%{text}<br>Metric: %{x}<br>Score: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(height=420)
        if plotly_events:
            clicked = plotly_events(fig, click_event=True, key="heatmap_click")
            if clicked:
                point = clicked[0]
                section_title = point.get("y")
                if section_title:
                    section = next((item for item in analysis.sections if item.title == section_title), None)
                    if section:
                        st.session_state["selected_section_id"] = section.id
        else:
            st.plotly_chart(fig, use_container_width=True)


def render_qa_tab(analysis: AnalysisResult) -> None:
    st.markdown("#### Q&A Workbench (Grounded)")

    for idx, turn in enumerate(st.session_state.get("qa_history", []), start=1):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("supported"):
                st.caption(f"Confidence score: {turn.get('score', 0.0):.3f}")
                for c_idx, citation in enumerate(turn.get("citations", []), start=1):
                    st.caption(f"[{c_idx}] {citation.section_title}: {citation.snippet}")
                    if st.button(
                        f"Jump to cited section {c_idx}",
                        key=f"qa_jump_{idx}_{c_idx}",
                    ):
                        jump_to_citation(citation)
            else:
                st.warning("Not found in document")
                nearest = turn.get("nearest", [])
                if nearest:
                    st.caption("Nearest relevant sections:")
                    for n_idx, citation in enumerate(nearest, start=1):
                        st.caption(f"- {citation.section_title}: {citation.snippet}")
                        if st.button(
                            f"Jump to nearest section {n_idx}",
                            key=f"qa_near_{idx}_{n_idx}",
                        ):
                            jump_to_citation(citation)

    prompt = st.chat_input("Ask a question about this document")
    if prompt:
        response = answer_question(prompt, analysis)
        st.session_state["qa_history"].append(
            {
                "question": prompt,
                "answer": response.answer,
                "supported": response.supported,
                "score": response.score,
                "citations": response.citations,
                "nearest": response.nearest_sections,
            }
        )
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Document Intelligence Dashboard",
        page_icon="📘",
        layout="wide",
    )
    init_state()

    st.title("Document Intelligence Dashboard")
    st.caption("Dynamic deep-dive interface for DOCX/PDF/TXT documents with grounded exploration tools.")

    with st.sidebar:
        st.header("Document Input")
        uploaded_file = st.file_uploader("Upload DOCX, PDF, or TXT", type=["docx", "pdf", "txt"])
        st.caption(f"Default path: `{DEFAULT_DOC_PATH}`")

    analysis, source_name = get_analysis(uploaded_file)
    if analysis is None:
        st.warning("Upload a document or place the default file at the configured path.")
        st.stop()

    if analysis.warnings:
        with st.expander("Pipeline Warnings"):
            for warning in analysis.warnings:
                st.write(f"- {warning}")

    st.success(
        f"Loaded `{analysis.file_name}` ({analysis.file_type.upper()}) | "
        f"Sections: {len(analysis.sections)} | Entities: {len(analysis.entities)} | "
        f"Requirements: {len(analysis.requirements)}"
    )

    command_palette_ui(analysis)
    if st.session_state.get("command_notice"):
        st.info(st.session_state["command_notice"])

    section_by_id, block_by_id, entities_by_section = section_maps(analysis)
    ensure_selected_section(analysis)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview",
            "Outline Navigator",
            "Entities & Relationships",
            "Requirements / Rubric Builder",
            "Visual Insights",
            "Q&A Workbench",
        ]
    )

    with tab1:
        render_overview_tab(analysis)
    with tab2:
        render_outline_tab(analysis, section_by_id, block_by_id, entities_by_section)
    with tab3:
        render_entities_tab(analysis)
    with tab4:
        render_requirements_tab(analysis)
    with tab5:
        render_visual_insights_tab(analysis)
    with tab6:
        render_qa_tab(analysis)


if __name__ == "__main__":
    main()
