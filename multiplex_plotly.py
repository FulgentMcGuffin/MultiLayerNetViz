"""Plotly 3D multiplex figure from Polars node/edge tables."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.colors import to_hex

from multi_layer_NetViz_fcts import EDGE_CMAP, PLANE_COLORS, _vivid_node_color


def _issuer_xy(issuers: list[str]) -> dict[str, tuple[float, float]]:
    n = max(len(issuers), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return {s: (float(np.cos(a)), float(np.sin(a))) for s, a in zip(issuers, angles)}

N_EDGE_BINS = 8
PLANE_PAD = 0.25
LAYER_GAP = 5.4


def _weight_limits(intra: pl.DataFrame) -> tuple[float, float]:
    if intra.is_empty() or "weight" not in intra.columns:
        return 0.0, 1.0
    weights = intra.get_column("weight").to_numpy()
    wmin, wmax = np.percentile(weights, [10, 90])
    if wmax <= wmin:
        wmin, wmax = float(weights.min()), float(weights.max())
    return float(wmin), float(wmax)


def _edge_hex(weight: float, wmin: float, wmax: float) -> str:
    span = wmax - wmin if wmax > wmin else 1.0
    t = float(np.clip((weight - wmin) / span, 0.0, 1.0))
    return to_hex(EDGE_CMAP(t))


def _bin_index(weight: float, wmin: float, wmax: float) -> int:
    span = wmax - wmin if wmax > wmin else 1.0
    t = float(np.clip((weight - wmin) / span, 0.0, 1.0))
    return min(int(t * N_EDGE_BINS), N_EDGE_BINS - 1)


def build_multiplex_figure(
    nodes: pl.DataFrame,
    intra: pl.DataFrame,
    inter: pl.DataFrame,
    visible_terms: list[str],
    *,
    all_issuers: list[str] | None = None,
    layer_gap: float = LAYER_GAP,
    title: str = "Yield-curve multiplex",
) -> go.Figure:
    terms = [t for t in visible_terms if t]
    if all_issuers is None:
        all_issuers = (
            sorted(nodes.get_column("issuer").unique().to_list())
            if not nodes.is_empty()
            else []
        )
    xy_of = _issuer_xy(all_issuers)
    z_of = {term: float(i) * layer_gap for i, term in enumerate(terms)}
    wmin, wmax = _weight_limits(intra)

    fig = go.Figure()
    if not terms:
        fig.update_layout(title=title, template="plotly_white")
        return fig

    xs = [xy_of[s][0] for s in all_issuers if s in xy_of]
    ys = [xy_of[s][1] for s in all_issuers if s in xy_of]
    xmin = (min(xs) if xs else -1.0) - PLANE_PAD
    xmax = (max(xs) if xs else 1.0) + PLANE_PAD
    ymin = (min(ys) if ys else -1.0) - PLANE_PAD
    ymax = (max(ys) if ys else 1.0) + PLANE_PAD

    for i, term in enumerate(terms):
        color = PLANE_COLORS[i % len(PLANE_COLORS)]
        z = z_of[term]
        fig.add_trace(
            go.Mesh3d(
                x=[xmin, xmax, xmax, xmin],
                y=[ymin, ymin, ymax, ymax],
                z=[z, z, z, z],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.38,
                hoverinfo="skip",
                showlegend=False,
                name=f"plane-{term}",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[xmin],
                y=[ymin],
                z=[z],
                mode="text",
                text=[term],
                textfont=dict(family="Arial", size=11, color="#333"),
                hoverinfo="skip",
                showlegend=False,
                name=f"label-{term}",
            )
        )

    issuer_index = {s: idx for idx, s in enumerate(all_issuers)}

    for term in terms:
        layer_nodes = nodes.filter(pl.col("term") == term) if not nodes.is_empty() else nodes
        if layer_nodes.is_empty():
            continue
        issuers = layer_nodes.get_column("issuer").to_list()
        x = [xy_of[s][0] for s in issuers]
        y = [xy_of[s][1] for s in issuers]
        z = [z_of[term]] * len(issuers)
        marker_colors = [
            _vivid_node_color(issuer_index.get(s, 0)) for s in issuers
        ]
        custom = [[ "node", s, term] for s in issuers]
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                text=issuers,
                textposition="top center",
                textfont=dict(family="Arial", size=8, color="#111"),
                marker=dict(
                    size=6,
                    color=marker_colors,
                    line=dict(color="black", width=1),
                    opacity=1.0,
                ),
                customdata=custom,
                hovertemplate="<b>%{text}</b><br>term=%{customdata[2]}<extra></extra>",
                name=term,
                legendgroup=term,
            )
        )

        layer_intra = intra.filter(pl.col("term") == term) if not intra.is_empty() else intra
        if layer_intra.is_empty():
            continue
        bins: list[dict] = [
            {"x": [], "y": [], "z": [], "mx": [], "my": [], "mz": [], "cd": [], "col": []}
            for _ in range(N_EDGE_BINS)
        ]
        for row in layer_intra.iter_rows(named=True):
            s, t = row["source_issuer"], row["target_issuer"]
            if s not in xy_of or t not in xy_of:
                continue
            w = float(row["weight"])
            b = _bin_index(w, wmin, wmax)
            x0, y0 = xy_of[s]
            x1, y1 = xy_of[t]
            zz = z_of[term]
            bins[b]["x"].extend([x0, x1, None])
            bins[b]["y"].extend([y0, y1, None])
            bins[b]["z"].extend([zz, zz, None])
            bins[b]["mx"].append(0.5 * (x0 + x1))
            bins[b]["my"].append(0.5 * (y0 + y1))
            bins[b]["mz"].append(zz)
            bins[b]["cd"].append(
                ["intra", s, t, term, w]
            )
            bins[b]["col"].append(_edge_hex(w, wmin, wmax))

        for b, bucket in enumerate(bins):
            if not bucket["x"]:
                continue
            t_mid = (b + 0.5) / N_EDGE_BINS
            line_color = to_hex(EDGE_CMAP(t_mid))
            fig.add_trace(
                go.Scatter3d(
                    x=bucket["x"],
                    y=bucket["y"],
                    z=bucket["z"],
                    mode="lines",
                    line=dict(color=line_color, width=3),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"intra-lines-{term}-{b}",
                    legendgroup=term,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=bucket["mx"],
                    y=bucket["my"],
                    z=bucket["mz"],
                    mode="markers",
                    marker=dict(size=4, color=bucket["col"], opacity=0.85),
                    customdata=bucket["cd"],
                    hovertemplate=(
                        "intra %{customdata[1]}–%{customdata[2]}"
                        "<br>term=%{customdata[3]}"
                        "<br>weight=%{customdata[4]:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"intra-picks-{term}-{b}",
                    legendgroup=term,
                )
            )

    # Real inter-layer edges from the graph, plus identity links between
    # consecutive *displayed* terms (same as the static matplotlib view).
    ix, iy, iz = [], [], []
    mx, my, mz, mcol, mcd = [], [], [], [], []

    def _add_inter_segment(s, t, tf, tt, w, extra_kind="inter"):
        if s not in xy_of or t not in xy_of:
            return
        if tf not in z_of or tt not in z_of:
            return
        x0, y0 = xy_of[s]
        x1, y1 = xy_of[t]
        z0, z1 = z_of[tf], z_of[tt]
        ix.extend([x0, x1, None])
        iy.extend([y0, y1, None])
        iz.extend([z0, z1, None])
        mx.append(0.5 * (x0 + x1))
        my.append(0.5 * (y0 + y1))
        mz.append(0.5 * (z0 + z1))
        mcol.append("#888888")
        mcd.append([extra_kind, s, t, tf, tt, w])

    if not inter.is_empty():
        for row in inter.iter_rows(named=True):
            _add_inter_segment(
                row["source_issuer"],
                row["target_issuer"],
                row["term_from"],
                row["term_to"],
                float(row["weight"]),
            )

    for i in range(len(terms) - 1):
        t0, t1 = terms[i], terms[i + 1]
        issuers_0 = set(
            nodes.filter(pl.col("term") == t0).get_column("issuer").to_list()
        ) if not nodes.is_empty() else set()
        issuers_1 = set(
            nodes.filter(pl.col("term") == t1).get_column("issuer").to_list()
        ) if not nodes.is_empty() else set()
        for issuer in sorted(issuers_0 & issuers_1):
            _add_inter_segment(issuer, issuer, t0, t1, 1.0, extra_kind="inter")

    if ix:
        fig.add_trace(
            go.Scatter3d(
                x=ix,
                y=iy,
                z=iz,
                mode="lines",
                line=dict(color="#888888", width=2),
                hoverinfo="skip",
                showlegend=False,
                name="inter-lines",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                marker=dict(size=4, color=mcol, opacity=0.8),
                customdata=mcd,
                hovertemplate=(
                    "inter %{customdata[1]}–%{customdata[2]}"
                    "<br>%{customdata[3]} → %{customdata[4]}"
                    "<br>weight=%{customdata[5]:.3f}<extra></extra>"
                ),
                showlegend=False,
                name="inter-picks",
            )
        )

    z_max = max(z_of.values()) if z_of else 1.0
    fig.update_layout(
        title=dict(text=title, font=dict(family="Arial", size=16), x=0.02, xanchor="left"),
        font=dict(family="Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.7),
            camera=dict(eye=dict(x=1.55, y=1.55, z=0.85)),
        ),
    )
    # Dummy trace so the blue-white-red colorbar is shown.
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[wmin],
                cmin=wmin,
                cmax=wmax,
                colorscale=[[0.0, "#2166ac"], [0.5, "#ffffff"], [1.0, "#b2182b"]],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Intra-layer<br>connection strength",
                        font=dict(family="Arial", size=11),
                    ),
                    thickness=14,
                    len=0.5,
                    x=1.01,
                    tickfont=dict(family="Arial", size=10),
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
            name="colorbar",
        )
    )
    fig.update_layout(scene_zaxis_range=[-0.35 * layer_gap, z_max + 0.35 * layer_gap])
    return fig
