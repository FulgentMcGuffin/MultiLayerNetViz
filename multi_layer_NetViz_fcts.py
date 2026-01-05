import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.path import Path
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Path3DCollection
import matplotlib.markers as mmarkers
from collections import defaultdict

plt.rcParams.update({'text.usetex': True})

def mscatter3d(x, y, z, ax=None, m=None, **kw):
    """
    Vectorized 3D scatter with multiple marker types.
    Groups points by marker to minimize draw calls.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if m is None:
        m = ['o'] * len(x)

    colors = kw.pop('c', None)
    if colors is None:
        colors = ['C0'] * len(x)
    elif isinstance(colors, str) or not hasattr(colors, '__len__'):
        colors = [colors] * len(x)

    sizes = kw.pop('s', None)
    if sizes is None:
        sizes = [100] * len(x)
    elif not hasattr(sizes, '__len__'):
        sizes = [sizes] * len(x)

    marker_groups = defaultdict(lambda: {'x': [], 'y': [], 'z': [], 'c': [], 's': []})
    for xi, yi, zi, mi, ci, si in zip(x, y, z, m, colors, sizes):
        g = marker_groups[mi]
        g['x'].append(xi)
        g['y'].append(yi)
        g['z'].append(zi)
        g['c'].append(ci)
        g['s'].append(si)

    scatters = []
    for marker, group in marker_groups.items():
        scat = ax.scatter(
            group['x'], group['y'], group['z'],
            marker=marker, c=group['c'], s=group['s'], **kw
        )
        scatters.append(scat)

    return scatters

def mscatter(x,y,ax=None, m=None, **kw):
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
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
    def __init__(self, graph, z_pos=None, label=None, label_pos=None, layout=nx.spring_layout,
                 node_size=50, node_color='C0', edge_color='dimgrey', node_marker='o', node_2d_marker='o',
                 node_label=None, edge_label=None, node_alpha=1.0, edge_alpha=0.5):
        self.graph = graph
        self.z_pos = z_pos
        self.label = label
        self.label_pos = label_pos
        self.layout = layout
        self.node_positions = None

        n_nodes = len(graph.nodes)
        self.node_size = self._expand_param(node_size, n_nodes)
        self.node_color = self._expand_param(node_color, n_nodes)
        self.node_marker = self._expand_param(node_marker, n_nodes)
        self.node_2d_marker = self._expand_param(node_2d_marker, n_nodes)
        self.node_alpha = self._expand_param(node_alpha, n_nodes)

        n_edges = len(graph.edges)
        self.edge_alpha = self._expand_param(edge_alpha, n_edges)
        self.edge_color = self._expand_param(edge_color, n_edges)
        self.node_label = node_label if node_label is not None else {}
        self.edge_label = edge_label if edge_label is not None else {}

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
        if ax:
            self.ax = ax
        else:
            fig = plt.figure(figsize=(10, 10))
            self.ax = fig.add_subplot(111, projection='3d')

    def draw(self):
        """Draws the entire multi-layer network."""
        for i,layer in enumerate(self.layers):
            if layer.z_pos is None:
                layer.z_pos = i # Default z position based on the layer index
            layer.compute_layout()
        for layer in self.layers:
            self._draw_plane(layer)
            self._draw_nodes(layer)
            self._draw_edges_within_layer(layer)
            self._draw_node_labels(layer)
            self._draw_edge_labels(layer)

        self._draw_edges_between_layers()
        self.ax.set_axis_off()

    def _draw_nodes(self, layer):
        nodes = list(layer.graph.nodes())
        x, y, z = zip(*[layer.node_positions[node] for node in nodes])

        # Map node properties from dicts if they are dicts
        node_colors = [layer.node_color[node] if isinstance(layer.node_color, dict) else layer.node_color[i] for i, node in enumerate(nodes)]
        node_sizes = [layer.node_size[node] if isinstance(layer.node_size, dict) else layer.node_size[i] for i, node in enumerate(nodes)]
        node_markers = [layer.node_marker[node] if isinstance(layer.node_marker, dict) else layer.node_marker[i] for i, node in enumerate(nodes)]
        node_alphas = [layer.node_alpha[node] if isinstance(layer.node_alpha, dict) else layer.node_alpha[i] for i, node in enumerate(nodes)]

        mscatter3d(x, y, z, ax=self.ax, m=node_markers, c=node_colors, s=node_sizes, alpha=node_alphas, zorder=99)

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

        line_collection = Line3DCollection(segments, color='dimgrey', alpha=0.1, linestyle='--', zorder=1)
        self.ax.add_collection3d(line_collection)

    def _draw_node_labels(self, layer):
        for node, label in layer.node_label.items():
            if node in layer.node_positions:
                color_text = 'black' if np.mean(mcolors.to_rgb(layer.node_color[node])) > 0.5 else 'white'
                self.ax.text(*layer.node_positions[node], label,
                             horizontalalignment='center', verticalalignment='center', color=color_text, zorder=100)

    def _draw_edge_labels(self, layer):
        for (u, v), label in layer.edge_label.items():
            pos_u = layer.node_positions[u]
            pos_v = layer.node_positions[v]
            pos_middle = [(3 * pos_u[i] + pos_v[i]) / 4 for i in range(3)]
            self.ax.text(*pos_middle, label)

    def _get_extent(self, pad=0.1):
        all_pos = np.array([pos for layer in self.layers for pos in layer.node_positions.values()])
        xmin, ymin, _ = np.min(all_pos, axis=0)
        xmax, ymax, _ = np.max(all_pos, axis=0)
        dx = xmax - xmin if xmax > xmin else 1
        dy = ymax - ymin if ymax > ymin else 1
        return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)

    def _draw_plane(self, layer, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self._get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u, v)
        W = layer.z_pos * np.ones_like(U)
        self.ax.plot_surface(U, V, W, alpha=0.1, zorder=1, *args, **kwargs)
        if layer.label:
            label_pos = layer.label_pos
            if label_pos is not None:
                xmin, ymin = label_pos
            self.ax.text(xmin, ymin, layer.z_pos, layer.label,
                         horizontalalignment='left', verticalalignment='top', fontsize=14)

    def draw_2d(self, ax=None):
        """Draws all layers flattened into a single 2D plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

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

        ax.set_axis_off()

def compute_node_colors(alignment, colormap=plt.cm.tab10):
    """
    Computes node colors for two layers based on an alignment matrix.

    Args:
        alignment (np.array): N1xN2 matrix aligning nodes from layer 1 to layer 2.
        colormap: Matplotlib colormap.

    Returns:
        tuple: (g1_colors, g2_colors)
    """
    N1, N2 = alignment.shape
    norm = Normalize(vmin=0, vmax=N2 - 1)
    g2_colors = [colormap(norm(i)) for i in range(N2)]
    g1_colors = np.dot(alignment, np.array([mcolors.to_rgb(c) for c in g2_colors]))

    return g1_colors, g2_colors
