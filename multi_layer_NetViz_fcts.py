


# MultiLayerNetViz: A Python library for visualizing multi-layer networks in 3D and 2D. By Gaspard Abel, 2025.
# SPDX-License-Identifier: CC-BY-4.0
#
# This work is licensed under the Creative Commons Attribution 4.0
# International License (CC BY 4.0).
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by/4.0/
# Adapted from https://stackoverflow.com/a/60416989/2912349. Published by paulbrodersen under CC BY 4.0

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.path import Path
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Path3DCollection
import matplotlib.markers as mmarkers
from collections import defaultdict


# Use matplotlib mathtext (Computer Modern) instead of system LaTeX.
# `text.usetex=True` requires a working latex + dvipng toolchain and fails
# if MiKTeX tries to auto-install packages during matplotlib's font probe.
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 12,
    'mathtext.fontset': 'cm',
})

def mscatter3d(x, y, z, ax=None, m=None, vmin=None, vmax=None, **kw):
    """
    Vectorized 3D scatter with multiple marker types.
    Groups points by marker to minimize draw calls.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    n_points = len(x)
    if m is None:
        m = ['o'] * n_points
    elif isinstance(m, str) or not hasattr(m, '__len__'):
        m = [m] * n_points
    elif len(m) != n_points:
        raise ValueError("Length of marker list must match number of points")

    colors = kw.pop('c', None)
    if colors is None:
        colors = ['C0'] * n_points
    elif isinstance(colors, str) or not hasattr(colors, '__len__'):
        colors = [colors] * n_points
    elif len(colors) != n_points:
        raise ValueError("Length of color list must match number of points")

    sizes = kw.pop('s', None)
    if sizes is None:
        sizes = [100] * n_points
    elif not hasattr(sizes, '__len__'):
        sizes = [sizes] * n_points
    elif len(sizes) != n_points:
        raise ValueError("Length of size list must match number of points")

    alphas = kw.pop('alpha', None)
    if alphas is None:
        alphas = [None] * n_points
    elif not hasattr(alphas, '__len__') or isinstance(alphas, str):
        alphas = [alphas] * n_points
    elif len(alphas) != n_points:
        raise ValueError("Length of alpha list must match number of points")

    edgecolors = kw.pop('edgecolors', kw.pop('edgecolor', None))
    if edgecolors is None:
        edgecolors = ['none'] * n_points
    elif isinstance(edgecolors, str) or not hasattr(edgecolors, '__len__'):
        edgecolors = [edgecolors] * n_points
    elif len(edgecolors) != n_points:
        raise ValueError("Length of edgecolors must match number of points")

    marker_groups = defaultdict(lambda: {'x': [], 'y': [], 'z': [], 'c': [], 's': [], 'm': [], 'a': [], 'e': []})
    for xi, yi, zi, mi, ci, si, ai, ei in zip(x, y, z, m, colors, sizes, alphas, edgecolors):
        g = marker_groups[mi]
        g['x'].append(xi)
        g['y'].append(yi)
        g['z'].append(zi)
        g['m'].append(mi)
        g['c'].append(ci)
        g['s'].append(si)
        g['a'].append(ai)
        g['e'].append(ei)

    scatters = []
    for marker, group in marker_groups.items():
        unique_alpha = set(group['a'])
        group_alpha = group['a'][0] if len(unique_alpha) == 1 else group['a']
        scat = ax.scatter(
            group['x'], group['y'], group['z'],
            marker=group['m'][0], c=group['c'], s=group['s'], alpha=group_alpha,
            edgecolors=group['e'], vmin=vmin, vmax=vmax, **kw
        )
        scatters.append(scat)

    return scatters

def mscatter(x,y,ax=None, m=None, vmin=None, vmax=None, **kw):
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y, vmin=vmin, vmax=vmax, **kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

# Display the nodes with the right perspective
def create_parallelogram_marker(angle_deg=0):
    # Base parallelogram vertices (centered at origin)
    base_vertices = np.array([
        [-0.1, -0.5], # Bottom left
        [ 0.9, -0.5], # Bottom right
        [ 0.3,  0.5], # Top right
        [-0.7,  0.5], # Top left
        [-0.1, -0.5]  # Close the shape
    ])

    # Convert angle to radians and create rotation matrix
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    # Rotate vertices
    rotated_vertices = base_vertices @ rotation_matrix.T

    # Create path
    return Path(rotated_vertices, closed=True)


def create_ellipse_marker(angle_deg=0, num_points=100):
    """Creates a rotated ellipse marker for plotting."""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 * np.cos(t)
    y = 0.25 * np.sin(t)
    verts = np.column_stack((x, y))

    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    rotated_verts = verts @ R.T
    return Path(rotated_verts, closed=True)

class Layer:
    """Represents a single layer in the multi-layer network."""
    def __init__(self, graph, z_pos=None, label=None, label_pos=None, label_zdir=None, layout=nx.spring_layout,
                 node_size=50, node_color='C0', edge_color='dimgrey', node_marker='o', node_2d_marker='o',
                 node_label=None, edge_label=None, node_alpha=1.0, edge_alpha=0.5, edge_arrows=None,
                 plane_alpha=0.1, plane_color=None, text_color='black', label_fontsize=12,
                 plane_label_fontsize=14, label_fontfamily=None,
                 node_edgecolor='none', node_edgewidth=0.0):
        self.graph = graph
        self.z_pos = z_pos
        self.label = label
        self.label_pos = label_pos
        self.label_zdir = label_zdir
        self.layout = layout
        self.node_positions = None

        n_nodes = len(graph.nodes)
        self.node_size = self._expand_param(node_size, n_nodes)
        self.node_color = self._expand_param(node_color, n_nodes)
        self.node_marker = self._expand_param(node_marker, n_nodes)
        self.node_2d_marker = self._expand_param(node_2d_marker, n_nodes)
        self.node_alpha = self._expand_param(node_alpha, n_nodes)
        self.node_edgecolor = self._expand_param(node_edgecolor, n_nodes)
        self.node_edgewidth = self._expand_param(node_edgewidth, n_nodes)

        n_edges = len(graph.edges)
        self.edge_alpha = self._expand_param(edge_alpha, n_edges)
        self.edge_color = self._expand_param(edge_color, n_edges)
        self.plane_alpha = self._expand_param(plane_alpha, 1)[0]
        self.plane_color = plane_color
        self.label_fontsize = label_fontsize
        self.plane_label_fontsize = plane_label_fontsize
        self.label_fontfamily = label_fontfamily

        self.node_label = node_label if node_label is not None else {}
        self.edge_label = edge_label if edge_label is not None else {}
        self.edge_arrows = edge_arrows if edge_arrows is not None else [(u, v, False) for u, v in graph.edges()]
        self.text_color = self._expand_param(text_color, n_nodes)

        self.node_positions = self.compute_layout()
    def _expand_param(self, param, n_items):
        if hasattr(param, '__len__') and not isinstance(param, str) and len(param) == n_items:
            return param
        return [param] * n_items

    def compute_layout(self, *args, **kwargs):
        """Computes the 2D layout for the layer's nodes."""
        pos = self.layout(self.graph, *args, **kwargs)
        self.node_positions = {node: (*coords, self.z_pos) for node, coords in pos.items()}

class LayeredNetworkGraph:
    """Plots a multi-layer network in 3D."""
    def __init__(self, layers, interlayer_edges=None, ax=None):
        self.layers = layers
        self.interlayer_edges = interlayer_edges if interlayer_edges is not None else []
        self.ax = ax

    def _ensure_3d_ax(self):
        """Create a 3D axes only when drawing in 3D, so 2D-only cells do not emit an extra figure."""
        if self.ax is None:
            fig = plt.figure(figsize=(10, 10))
            self.ax = fig.add_subplot(111, projection='3d')
        return self.ax

    def draw(self):
        """Draws the entire multi-layer network."""
        self._ensure_3d_ax()
        for i,layer in enumerate(self.layers):
            if layer.z_pos is None:
                layer.z_pos = i # Default z position based on the layer index
            layer.compute_layout()
        # start from the lowest zpos
        for layer in sorted(self.layers, key=lambda l: l.z_pos):
            self._draw_plane(layer)
            self._draw_nodes(layer)
            self._draw_edges_within_layer(layer)
            self._draw_node_labels(layer)
            self._draw_edge_arrows(layer)
            self._draw_edge_labels(layer)

        self._draw_edges_between_layers()
        # self.ax.set_axis_off()

    def _draw_nodes(self, layer):
        nodes = list(layer.graph.nodes())
        x, y, z = zip(*[layer.node_positions[node] for node in nodes])

        # Map node properties from dicts if they are dicts
        node_colors = [layer.node_color[node] if isinstance(layer.node_color, dict) else layer.node_color[i] for i, node in enumerate(nodes)]
        node_sizes = [layer.node_size[node] if isinstance(layer.node_size, dict) else layer.node_size[i] for i, node in enumerate(nodes)]
        node_markers = [layer.node_marker[node] if isinstance(layer.node_marker, dict) else layer.node_marker[i] for i, node in enumerate(nodes)]
        node_alphas = [layer.node_alpha[node] if isinstance(layer.node_alpha, dict) else layer.node_alpha[i] for i, node in enumerate(nodes)]
        node_edgecolors = [layer.node_edgecolor[node] if isinstance(layer.node_edgecolor, dict) else layer.node_edgecolor[i] for i, node in enumerate(nodes)]

        mscatter3d(
            x, y, z, ax=self.ax, m=node_markers, c=node_colors, s=node_sizes,
            alpha=node_alphas, zorder=99,
            edgecolors=node_edgecolors, linewidths=layer.node_edgewidth,
            depthshade=False,
        )

    def _draw_edges_within_layer(self, layer):
        segments = [(layer.node_positions[u], layer.node_positions[v]) for u, v in layer.graph.edges()]
        line_collection = Line3DCollection(segments, color=layer.edge_color, alpha=layer.edge_alpha, linestyle='-', zorder=1)
        self.ax.add_collection3d(line_collection)

    def _draw_edges_between_layers(self):
        if not self.interlayer_edges:
            return

        segments = []
        for u_id, v_id, z1, z2 in self.interlayer_edges:
            pos1 = self.layers[z1].node_positions[u_id]
            pos2 = self.layers[z2].node_positions[v_id]
            segments.append((pos1, pos2))

        line_collection = Line3DCollection(segments, color='dimgrey', alpha=0.2, linestyle='--', zorder=1)
        self.ax.add_collection3d(line_collection)

    def _draw_node_labels(self, layer):
        for node, label in layer.node_label.items():
            if node in layer.node_positions:
                # check if there is a custom color for the text, if not use black or white depending on the node color
                if isinstance(layer.text_color, dict) and node in layer.text_color:
                    color_text = layer.text_color[node] if isinstance(layer.text_color, dict) else layer.text_color
                else:
                    color_text = 'black' if np.mean(mcolors.to_rgb(layer.node_color[node])) > 0.5 else 'white'
                text_kw = dict(
                    horizontalalignment='center', verticalalignment='center',
                    color=color_text, zorder=100, fontsize=layer.label_fontsize,
                )
                if layer.label_fontfamily:
                    text_kw['fontfamily'] = layer.label_fontfamily
                self.ax.text(*layer.node_positions[node], label, **text_kw)

    def _draw_edge_labels(self, layer):
        for (u, v), label in layer.edge_label.items():
            pos_u = layer.node_positions[u]
            pos_v = layer.node_positions[v]
            pos_middle = [(3 * pos_u[i] + pos_v[i]) / 4 for i in range(3)]
            diff_pos = np.array(pos_v) - np.array(pos_u)
            # Project onto XY-plane (set z to 0)
            diff_pos_plane = np.array([diff_pos[0], diff_pos[1], 0.0])

            # Normalize (to avoid length distortion)
            norm = np.linalg.norm(diff_pos_plane)
            if norm > 0:
                diff_pos_plane /= norm

            # place the text above the arrow, projected and rotated in its direction
            text_pos = [pos_middle[0] + 0.05*diff_pos_plane[0],
                        pos_middle[1] + 0.05*diff_pos_plane[1],
                        pos_middle[2] + 0.03]
            text_pos = pos_v + 0.05*diff_pos_plane
            self.ax.text(*text_pos, label, verticalalignment='center', horizontalalignment='center',  zdir = [diff_pos[0], diff_pos[1], 0.0],zorder=100)

    def _draw_edge_arrows(self, layer):
        for (u, v, do_arrow) in layer.edge_arrows:
            if do_arrow:
                # in that case draw a small arrow in the direction of the target
                pos_target = layer.node_positions[v]
                pos_source = layer.node_positions[u]
                pos_middle = [(3*pos_source[i] + pos_target[i]) / 4 for i in range(3)]
                diff_pos = np.array(pos_target) - np.array(pos_source)
                # Project onto XY-plane (set z to 0)
                diff_pos_plane = np.array([diff_pos[0], diff_pos[1], 0.0])

                # Normalize (to avoid length distortion)
                norm = np.linalg.norm(diff_pos_plane)
                if norm > 0:
                    diff_pos_plane /= norm
                self.ax.quiver(*pos_middle, *diff_pos_plane, length=0.1, normalize=False,
                            color='black', arrow_length_ratio=0.3,zorder=101)

    def _get_extent(self, pad=0.1):
        all_pos = np.array([pos for layer in self.layers for pos in layer.node_positions.values()])
        xmin, ymin, _ = np.min(all_pos, axis=0)
        xmax, ymax, _ = np.max(all_pos, axis=0)
        dx = xmax - xmin if xmax > xmin else 1
        dy = ymax - ymin if ymax > ymin else 1
        return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)

    def _draw_plane(self, layer, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self._get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 5)
        v = np.linspace(ymin, ymax, 5)
        U, V = np.meshgrid(u, v)
        W = layer.z_pos * np.ones_like(U)
        # if layer.plane_alpha exists, use it, otherwise default to 0.1
        alpha = getattr(layer, 'plane_alpha', 0.1)
        plane_color = getattr(layer, 'plane_color', None)
        # 3D plot_surface often ignores `color=` and maps Z through a colormap
        # (the yellow planes). Paint with an explicit RGBA facecolor array.
        surface_kw = dict(shade=False, linewidth=0, antialiased=True, zorder=1)
        if plane_color is not None:
            rgba = mcolors.to_rgba(plane_color, alpha=alpha)
            facecolors = np.empty(U.shape + (4,), dtype=float)
            facecolors[:] = rgba
            surface_kw['facecolors'] = facecolors
        else:
            surface_kw['alpha'] = alpha
        kwargs = {k: v for k, v in kwargs.items() if k not in ('color', 'cmap', 'facecolors', 'alpha')}
        surface_kw.update(kwargs)
        self.ax.plot_surface(U, V, W, *args, **surface_kw)
        if layer.label:
            label_pos = layer.label_pos
            if label_pos is not None:
                xmin, ymin = label_pos
            label_zdir = layer.label_zdir
            plane_text_kw = dict(
                horizontalalignment='left', verticalalignment='top',
                fontsize=getattr(layer, 'plane_label_fontsize', 14),
                zdir=label_zdir,
            )
            if getattr(layer, 'label_fontfamily', None):
                plane_text_kw['fontfamily'] = layer.label_fontfamily
            self.ax.text(xmin, ymin, layer.z_pos, layer.label, **plane_text_kw)

    def draw_2d(self, ax=None, interlayer_edges_2d=None):
        """Draws all layers flattened into a single 2D plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        # Do not leave a constructor/3D figure open; Jupyter would display it too.
        if self.ax is not None and self.ax is not ax:
            plt.close(self.ax.get_figure())

        for layer in self.layers:
            if layer.node_positions is None:
                layer.compute_layout()

            # Draw edges
            for i,uv in enumerate(layer.graph.edges()):
                u, v = uv
                pos_u = layer.node_positions[u]
                pos_v = layer.node_positions[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], color=layer.edge_color[i], alpha=0.7, linestyle='-', zorder=1)


            # Draw nodes
            nodes = list(layer.graph.nodes())
            x, y, _ = zip(*[layer.node_positions[node] for node in nodes])

            node_colors = [layer.node_color[node] if isinstance(layer.node_color, dict) else layer.node_color[i] for i, node in enumerate(nodes)]
            node_sizes = [layer.node_size[node] if isinstance(layer.node_size, dict) else layer.node_size[i] for i, node in enumerate(nodes)]
            node_markers = [layer.node_2d_marker[node] if isinstance(layer.node_2d_marker, dict) else layer.node_2d_marker[i] for i, node in enumerate(nodes)]
            node_alphas = [layer.node_alpha[node] if isinstance(layer.node_alpha, dict) else layer.node_alpha[i] for i, node in enumerate(nodes)]

            mscatter(x, y, ax=ax, m=node_markers, c=node_colors, s=node_sizes, alpha=node_alphas, zorder=99)

            # Draw node labels
            for node, label in layer.node_label.items():
                if node in layer.node_positions:
                    color_text = 'black' if np.mean(mcolors.to_rgb(layer.node_color[node])) > 0.5 else 'white'
                    ax.text(layer.node_positions[node][0], layer.node_positions[node][1], label,
                                 horizontalalignment='center', verticalalignment='center', color=color_text, zorder=100)

            # Draw edge labels
            for (u, v), label in layer.edge_label.items():
                pos_u = layer.node_positions[u]
                pos_v = layer.node_positions[v]

                diff_xy = np.array([pos_v[0] - pos_u[0], pos_v[1] - pos_u[1]], dtype=float)

                # Normalize in data space to place the label slightly after the target node.
                norm = np.linalg.norm(diff_xy)
                if norm > 0:
                    diff_xy /= norm

                text_pos = np.array([pos_v[0], pos_v[1]], dtype=float) + 0.05 * diff_xy

                # Compute angle in display coordinates so text follows the visual edge direction.
                p1 = ax.transData.transform((pos_u[0], pos_u[1]))
                p2 = ax.transData.transform((pos_v[0], pos_v[1]))
                rotation_text = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
                # if the angle is too steep, flip it to avoid upside-down text
                if rotation_text > 90:
                    rotation_text -= 180
                elif rotation_text < -90:
                    rotation_text += 180
                ax.text(
                    text_pos[0], text_pos[1], label,
                    rotation=rotation_text,
                    rotation_mode='anchor',
                    transform_rotates_text=False,
                    horizontalalignment='center',
                    verticalalignment='center',
                    zorder=100
                )
        # Draw interlayer edges in 2D
        if interlayer_edges_2d:
            for u_id, v_id, z1, z2 in interlayer_edges_2d:
                pos1 = self.layers[z1].node_positions[u_id]
                pos2 = self.layers[z2].node_positions[v_id]
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], color='dimgrey', alpha=0.2, linestyle='--', zorder=1)

        ax.set_axis_off()

def compute_node_colors(alignment, colormap=plt.cm.tab10, vmin=None, vmax=None):
    """
    Computes node colors for two layers based on an alignment matrix.

    Args:
        alignment (np.array): N1xN2 matrix aligning nodes from layer 1 to layer 2.
        colormap: Matplotlib colormap.
        vmin (float, optional): Minimum value for colormap normalization. Defaults to None.
        vmax (float, optional): Maximum value for colormap normalization. Defaults to None.

    Returns:
        tuple: (g1_colors, g2_colors)
    """
    N1, N2 = alignment.shape

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = N2 - 1

    norm = Normalize(vmin=vmin, vmax=vmax)
    g2_colors = [colormap(norm(i)) for i in range(N2)]
    g1_colors = np.dot(alignment, np.array([mcolors.to_rgb(c) for c in g2_colors]))

    return g1_colors, g2_colors


def adjacency_to_edges(
    adj: np.ndarray,
    *,
    threshold: float = 0.0,
    include_self_loops: bool = False,
) -> pl.DataFrame:
    """Convert a dense adjacency matrix into a Polars edge table.

    Returns columns ``source``, ``target``, and ``weight`` for entries strictly
    above ``threshold``. Linear-algebra layouts still use NumPy; this is for
    tabular edge filtering, labels, and arrows.
    """
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square 2D array")

    sources, targets = np.nonzero(adj > threshold)
    weights = adj[sources, targets]
    if not include_self_loops:
        keep = sources != targets
        sources, targets, weights = sources[keep], targets[keep], weights[keep]

    return pl.DataFrame({
        "source": sources.astype(np.int64, copy=False),
        "target": targets.astype(np.int64, copy=False),
        "weight": weights.astype(np.float64, copy=False),
    })


def node_activity(event_matrix: np.ndarray, *, axis: str) -> pl.DataFrame:
    """Sum a 2D event matrix into a Polars table of per-node activity.

    ``axis='columns'`` sums down each column (one total per column index).
    ``axis='rows'`` sums across each row (one total per row index).
    """
    frame = pl.DataFrame(np.asarray(event_matrix))
    if axis == "columns":
        totals = frame.sum().transpose(include_header=False, column_names=["activity"])
        return totals.with_row_index("id")
    if axis == "rows":
        return frame.select(pl.sum_horizontal(pl.all()).alias("activity")).with_row_index("id")
    raise ValueError("axis must be 'rows' or 'columns'")


def top_k_partners(
    scores: np.ndarray,
    k: int = 1,
    *,
    row_name: str = "row",
    col_name: str = "col",
    value_name: str = "score",
) -> pl.DataFrame:
    """Return the ``k`` highest-scoring row partners for each column.

    ``scores`` is a dense 2D array (rows × columns). The result is a long
    Polars table with columns ``row_name``, ``col_name``, and ``value_name``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array")

    n_cols = scores.shape[1]
    return (
        pl.DataFrame(scores, schema=[str(i) for i in range(n_cols)])
        .with_row_index(row_name)
        .unpivot(index=row_name, variable_name=col_name, value_name=value_name)
        .with_columns(pl.col(col_name).cast(pl.Int64))
        .sort(value_name, descending=True)
        .group_by(col_name, maintain_order=True)
        .head(k)
    )


PLANE_COLORS = ("#4c9ad6", "#5aa33a", "#e07050", "#8b62c4")  # cycle if more than 4 layers
EDGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_white_red",
    ["#2166ac", "#ffffff", "#b2182b"],
)


def _vivid_node_color(i: int) -> str:
    r, g, b, _ = plt.cm.tab10(i % 10)
    hsv = mcolors.rgb_to_hsv((r, g, b))
    hsv[1] = min(1.0, float(hsv[1]) * 1.25)
    hsv[2] = min(0.82, float(hsv[2]) * 0.88)
    return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))


def _shared_circular_layout(issuers: list[str]):
    n = max(len(issuers), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = {s: (float(np.cos(a)), float(np.sin(a))) for s, a in zip(issuers, angles)}

    def layout(G, *args, **kwargs):
        return {node: xy[node] for node in G.nodes()}

    return layout


def _intra_graph(G, term: str):
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data.get("layer") == "inter":
            continue
        if not (isinstance(u, tuple) and isinstance(v, tuple)):
            continue
        if u[1] != term or v[1] != term:
            continue
        H.add_edge(u[0], v[0], weight=float(data.get("weight", 0.5)))
    for n in G.nodes():
        if isinstance(n, tuple) and n[1] == term:
            H.add_node(n[0])
    return H


def _edge_weights(H) -> list[float]:
    return [float(d.get("weight", 0.5)) for _, _, d in H.edges(data=True)]


def _edge_colors(weights: list[float], vmin: float, vmax: float) -> list[str]:
    if not weights:
        return []
    span = vmax - vmin if vmax > vmin else 1.0
    cmap = EDGE_CMAP
    colors = []
    for w in weights:
        t = float(np.clip((w - vmin) / span, 0.0, 1.0))
        colors.append(mcolors.to_hex(cmap(t)))
    return colors


def plot_multilayer_netviz(
    G,
    terms: list[str],
    title: str = "Yield-curve multiplex",
    layer_gap: float = 5.4,
):
    """Draw a 3D multiplex with one Layer per term. Plane colors cycle after 4 layers."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.figure import Figure

    issuers = sorted({n[0] for n in G.nodes() if isinstance(n, tuple)})
    layout = _shared_circular_layout(issuers)
    graphs = [_intra_graph(G, term) for term in terms]
    all_w = np.asarray([w for H in graphs for w in _edge_weights(H)], dtype=float)
    if all_w.size:
        # Clip outliers so the colormap is spent on typical weights, not the extremes.
        wmin, wmax = np.percentile(all_w, [10, 90])
        if wmax <= wmin:
            wmin, wmax = float(all_w.min()), float(all_w.max())
    else:
        wmin, wmax = 0.0, 1.0

    layers = []
    for i, (term, H) in enumerate(zip(terms, graphs)):
        node_color = {s: _vivid_node_color(issuers.index(s)) for s in H.nodes()}
        layers.append(
            Layer(
                H,
                z_pos=float(i) * layer_gap,
                label=term,
                layout=layout,
                node_size=42,
                node_color=node_color,
                text_color={s: "black" for s in H.nodes()},
                edge_color=_edge_colors(_edge_weights(H), wmin, wmax) or "dimgrey",
                node_label={s: s for s in H.nodes()},
                node_alpha=1.0,
                edge_alpha=0.9,
                plane_alpha=0.5,
                plane_color=PLANE_COLORS[i % len(PLANE_COLORS)],
                label_fontsize=6,
                plane_label_fontsize=8,
                label_fontfamily="Arial",
                node_edgecolor="black",
                node_edgewidth=1.15,
            )
        )

    interlayer = []
    for i in range(len(terms) - 1):
        common = set(layers[i].graph.nodes()) & set(layers[i + 1].graph.nodes())
        for s in common:
            interlayer.append((s, s, i, i + 1))

    fig = Figure(figsize=(8.8, 7.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    viz = LayeredNetworkGraph(layers, interlayer_edges=interlayer, ax=ax)
    viz.draw()
    ax.set_title(title, fontsize=11, fontfamily="Arial", pad=2)
    ax.set_axis_off()
    z_span = max((len(terms) - 1) * layer_gap, 1.0)
    ax.set_box_aspect((1.0, 1.0, 2.0))
    ax.set_zlim(-0.35 * layer_gap, z_span + 0.35 * layer_gap)

    sm = ScalarMappable(cmap=EDGE_CMAP, norm=Normalize(vmin=wmin, vmax=wmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, fraction=0.03)
    cbar.set_label("Intra-layer connection strength", fontfamily="Arial", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily("Arial")
    fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.02)
    return fig, ax

