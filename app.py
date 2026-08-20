"""Interactive Dash app for the yield-curve multiplex."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from multiplex_data import available_terms, filter_tables, multiplex_tables
from multiplex_plotly import build_multiplex_figure

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "resources" / "data" / "multilayer_network.pkl"
DEFAULT_TERMS = ["Y001p0", "Y005p0", "Y010p0", "Y030p0"]

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Multiplex pickle not found at {DATA_PATH}. "
        "Place multilayer_network.pkl under resources/data/."
    )

with DATA_PATH.open("rb") as fh:
    _payload = pickle.load(fh)
_GRAPH = _payload["graph"]
NODES, INTRA, INTER = multiplex_tables(_GRAPH)
ALL_TERMS = available_terms(NODES)
ALL_ISSUERS = (
    sorted(NODES.get_column("issuer").unique().to_list()) if not NODES.is_empty() else []
)
DEFAULT_VISIBLE = [t for t in DEFAULT_TERMS if t in ALL_TERMS] or ALL_TERMS[:4]
TABLE_PAGE_SIZE = 8
DBLCLICK_MS = 450


def _records(df) -> list[dict]:
    if df is None or df.is_empty():
        return []
    return df.to_dicts()


def _columns(df) -> list[dict]:
    if df is None or df.is_empty():
        return []
    return [{"name": c, "id": c} for c in df.columns]


def _format_selection(customdata) -> str:
    if not customdata:
        return (
            "Click a node or an edge midpoint for details.\n"
            "Double-click to jump to that row in the tables below."
        )
    kind = customdata[0]
    if kind == "node":
        _, issuer, term = customdata[:3]
        return (
            f"Node\n"
            f"  issuer: {issuer}\n"
            f"  term:   {term}\n"
            f"  id:     ({issuer}, {term})"
        )
    if kind == "intra":
        _, src, tgt, term, weight = customdata[:5]
        return (
            f"Intra-layer edge\n"
            f"  {src} — {tgt}\n"
            f"  term:   {term}\n"
            f"  weight: {float(weight):.6f}"
        )
    if kind == "inter":
        _, src, tgt, term_from, term_to, weight = customdata[:6]
        return (
            f"Inter-layer edge\n"
            f"  {src} — {tgt}\n"
            f"  {term_from} → {term_to}\n"
            f"  weight: {float(weight):.6f}"
        )
    return json.dumps(customdata, indent=2, default=str)


def _undirected_match(a: str, b: str, c: str, d: str) -> bool:
    return (a == c and b == d) or (a == d and b == c)


def _find_intra_edge_row(rows: list[dict], src: str, tgt: str, term: str) -> int | None:
    for i, row in enumerate(rows or []):
        if row.get("term") != term:
            continue
        if _undirected_match(
            str(row.get("source_issuer")),
            str(row.get("target_issuer")),
            str(src),
            str(tgt),
        ):
            return i
    return None


def _find_inter_edge_row(
    rows: list[dict], src: str, tgt: str, term_from: str, term_to: str
) -> int | None:
    for i, row in enumerate(rows or []):
        issuers_ok = _undirected_match(
            str(row.get("source_issuer")),
            str(row.get("target_issuer")),
            str(src),
            str(tgt),
        )
        terms_ok = _undirected_match(
            str(row.get("term_from")),
            str(row.get("term_to")),
            str(term_from),
            str(term_to),
        )
        if issuers_ok and terms_ok:
            return i
    return None


def _find_first_node_row(
    intra_rows: list[dict], inter_rows: list[dict], issuer: str, term: str
) -> tuple[str, int | None]:
    """Return ('intra'|'inter', row_index) for the first table row involving this node."""
    for i, row in enumerate(intra_rows or []):
        if row.get("term") != term:
            continue
        if issuer in (str(row.get("source_issuer")), str(row.get("target_issuer"))):
            return "intra", i
    for i, row in enumerate(inter_rows or []):
        terms = {str(row.get("term_from")), str(row.get("term_to"))}
        issuers = {str(row.get("source_issuer")), str(row.get("target_issuer"))}
        if issuer in issuers and term in terms:
            return "inter", i
    return "intra", None


def _table_jump(idx: int | None, columns: list[dict]):
    if idx is None:
        return 0, [], None
    page = idx // TABLE_PAGE_SIZE
    col_id = columns[0]["id"] if columns else "source_issuer"
    active = {"row": idx % TABLE_PAGE_SIZE, "column": 0, "column_id": col_id}
    return page, [idx], active


_TABLE_SELECTED_STYLE = [
    {
        "if": {"state": "selected"},
        "backgroundColor": "#cfe8ff",
        "fontWeight": "bold",
    },
    {
        "if": {"state": "active"},
        "backgroundColor": "#fff3cd",
    },
]


app = Dash(__name__)
app.title = "Yield-curve multiplex"

app.layout = html.Div(
    style={
        "fontFamily": "Arial, Helvetica, sans-serif",
        "margin": "0",
        "padding": "12px 16px",
        "background": "#f7f7f7",
        "minHeight": "100vh",
    },
    children=[
        html.H2("Yield-curve multiplex", style={"margin": "0 0 8px 0"}),
        dcc.Store(id="dblclick-store"),
        html.Div(id="click-bind-dummy", style={"display": "none"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Networks (terms)", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="layer-checklist",
                            options=[{"label": t, "value": t} for t in ALL_TERMS],
                            value=DEFAULT_VISIBLE,
                            inline=False,
                            style={"maxHeight": "70vh", "overflowY": "auto"},
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"display": "block", "margin": "2px 0"},
                        ),
                    ],
                    style={
                        "width": "180px",
                        "flexShrink": "0",
                        "background": "white",
                        "padding": "10px",
                        "borderRadius": "6px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="multiplex-graph",
                            style={"height": "72vh"},
                            config={"displaylogo": False},
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0", "background": "white",
                           "borderRadius": "6px", "padding": "4px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"},
                ),
                html.Div(
                    [
                        html.Label("Selection", style={"fontWeight": "bold"}),
                        html.Pre(
                            id="selection-info",
                            style={
                                "whiteSpace": "pre-wrap",
                                "fontSize": "13px",
                                "margin": "8px 0 0 0",
                            },
                            children=_format_selection(None),
                        ),
                    ],
                    style={
                        "width": "260px",
                        "flexShrink": "0",
                        "background": "white",
                        "padding": "10px",
                        "borderRadius": "6px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
                    },
                ),
            ],
            style={"display": "flex", "gap": "12px", "alignItems": "stretch"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Intra-layer edges", style={"margin": "12px 0 6px"}),
                        dash_table.DataTable(
                            id="intra-table",
                            page_size=TABLE_PAGE_SIZE,
                            page_current=0,
                            selected_rows=[],
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "4px"},
                            style_header={"fontWeight": "bold"},
                            style_data_conditional=_TABLE_SELECTED_STYLE,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                ),
                html.Div(
                    [
                        html.H4("Inter-layer edges", style={"margin": "12px 0 6px"}),
                        dash_table.DataTable(
                            id="inter-table",
                            page_size=TABLE_PAGE_SIZE,
                            page_current=0,
                            selected_rows=[],
                            style_table={"overflowX": "auto"},
                            style_cell={"fontFamily": "Arial", "fontSize": "12px", "padding": "4px"},
                            style_header={"fontWeight": "bold"},
                            style_data_conditional=_TABLE_SELECTED_STYLE,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "marginTop": "12px"},
        ),
    ],
)


@app.callback(
    Output("multiplex-graph", "figure"),
    Output("intra-table", "data"),
    Output("intra-table", "columns"),
    Output("inter-table", "data"),
    Output("inter-table", "columns"),
    Output("intra-table", "page_current"),
    Output("intra-table", "selected_rows"),
    Output("intra-table", "active_cell"),
    Output("inter-table", "page_current"),
    Output("inter-table", "selected_rows"),
    Output("inter-table", "active_cell"),
    Input("layer-checklist", "value"),
)
def update_layers(visible_terms):
    terms = visible_terms or []
    nodes, intra, inter = filter_tables(NODES, INTRA, INTER, terms)
    fig = build_multiplex_figure(
        nodes, intra, inter, terms, all_issuers=ALL_ISSUERS
    )
    return (
        fig,
        _records(intra),
        _columns(intra),
        _records(inter),
        _columns(inter),
        0,
        [],
        None,
        0,
        [],
        None,
    )


@app.callback(
    Output("selection-info", "children"),
    Input("multiplex-graph", "clickData"),
    prevent_initial_call=True,
)
def update_selection(click_data):
    if not click_data or "points" not in click_data:
        return no_update
    point = click_data["points"][0]
    custom = point.get("customdata")
    return _format_selection(custom)


app.clientside_callback(
    """
    function(fig) {
        const el = document.querySelector('#multiplex-graph .js-plotly-plot');
        if (!el) {
            return window.dash_clientside.no_update;
        }
        if (el._muxHandler && el.removeListener) {
            el.removeListener('plotly_click', el._muxHandler);
        }
        let lastT = 0;
        let lastK = '';
        el._muxHandler = function(ev) {
            if (!ev || !ev.points || !ev.points.length) {
                return;
            }
            const cd = ev.points[0].customdata;
            if (!cd) {
                return;
            }
            const key = JSON.stringify(cd);
            const now = Date.now();
            if (now - lastT < 450 && key === lastK) {
                window.dash_clientside.set_props('dblclick-store', {
                    data: {customdata: cd, t: now}
                });
            }
            lastT = now;
            lastK = key;
        };
        el.on('plotly_click', el._muxHandler);
        return window.dash_clientside.no_update;
    }
    """,
    Output("click-bind-dummy", "children"),
    Input("multiplex-graph", "figure"),
)


@app.callback(
    Output("selection-info", "children", allow_duplicate=True),
    Output("intra-table", "page_current", allow_duplicate=True),
    Output("intra-table", "selected_rows", allow_duplicate=True),
    Output("intra-table", "active_cell", allow_duplicate=True),
    Output("inter-table", "page_current", allow_duplicate=True),
    Output("inter-table", "selected_rows", allow_duplicate=True),
    Output("inter-table", "active_cell", allow_duplicate=True),
    Input("dblclick-store", "data"),
    State("intra-table", "data"),
    State("intra-table", "columns"),
    State("inter-table", "data"),
    State("inter-table", "columns"),
    prevent_initial_call=True,
)
def jump_from_double_click(payload, intra_rows, intra_cols, inter_rows, inter_cols):
    if not payload or not payload.get("customdata"):
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    custom = payload["customdata"]
    info = _format_selection(custom)
    kind = custom[0] if custom else None

    intra_page, intra_sel, intra_active = 0, [], None
    inter_page, inter_sel, inter_active = 0, [], None

    if kind == "intra":
        _, src, tgt, term, _weight = custom[:5]
        idx = _find_intra_edge_row(intra_rows, src, tgt, term)
        intra_page, intra_sel, intra_active = _table_jump(idx, intra_cols or [])
        if idx is None:
            info += "\n\n(No matching intra-layer row in the current table.)"
    elif kind == "inter":
        _, src, tgt, term_from, term_to, _weight = custom[:6]
        idx = _find_inter_edge_row(inter_rows, src, tgt, term_from, term_to)
        inter_page, inter_sel, inter_active = _table_jump(idx, inter_cols or [])
        if idx is None:
            info += "\n\n(No matching inter-layer row in the current table.)"
    elif kind == "node":
        _, issuer, term = custom[:3]
        which, idx = _find_first_node_row(intra_rows, inter_rows, issuer, term)
        if which == "inter":
            inter_page, inter_sel, inter_active = _table_jump(idx, inter_cols or [])
        else:
            intra_page, intra_sel, intra_active = _table_jump(idx, intra_cols or [])
        if idx is None:
            info += "\n\n(No table row for this node in the visible networks.)"
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    return info, intra_page, intra_sel, intra_active, inter_page, inter_sel, inter_active


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
