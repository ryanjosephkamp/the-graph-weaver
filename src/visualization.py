"""
The Graph Weaver — Visualization Module.

Dual rendering engine for protein graph visualization:
    - PlotlyRenderer: Interactive 3-D graph and chart visualizations
    - MatplotlibRenderer: Static publication-quality figures
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from src.graph_engine import (
    ProteinStructure,
    ProteinGraph,
    NodeFeatures,
    EdgeData,
    AdjacencyData,
    GraphStatistics,
    Residue,
    AMINO_ACIDS,
    AA_INDEX,
    HYDROPHOBICITY,
    CHARGE,
    CONTACT_COLORS,
    CONTACT_DESCRIPTIONS,
    classify_contacts,
)
from src.analysis import (
    FullGraphAnalysis,
    KSweepAnalysis,
    ContactAnalysis,
    FeatureAnalysis,
    PresetComparisonResult,
)


# ═══════════════════════════════════════════════════════════════════════
# Color helpers
# ═══════════════════════════════════════════════════════════════════════

# Sequence distance → color (continuous)
def _seq_dist_color(seq_dist: int, max_dist: int = 50) -> str:
    """Map sequence distance to a color (blue → orange → red)."""
    t = min(seq_dist / max(max_dist, 1), 1.0)
    if t < 0.1:
        return "rgba(150,150,150,0.4)"   # backbone (grey)
    elif t < 0.2:
        return f"rgba(33,150,243,{0.3 + 0.5*t})"  # short-range (blue)
    elif t < 0.4:
        return f"rgba(255,152,0,{0.4 + 0.4*t})"   # medium (orange)
    else:
        return f"rgba(244,67,54,{0.5 + 0.5*t})"    # long-range (red)


def _contact_color(contact_type: str) -> str:
    """Get color for a contact type."""
    return CONTACT_COLORS.get(contact_type, "#888888")


# Node coloring by hydrophobicity
HYDRO_COLORSCALE = [
    [0.0, "#2196F3"],   # hydrophilic (blue)
    [0.5, "#FFFFFF"],   # neutral (white)
    [1.0, "#F44336"],   # hydrophobic (red)
]

# Node coloring by charge
CHARGE_COLORSCALE = [
    [0.0, "#F44336"],   # negative (red)
    [0.5, "#FFFFFF"],   # neutral (white)
    [1.0, "#2196F3"],   # positive (blue)
]


# ═══════════════════════════════════════════════════════════════════════
# PlotlyRenderer
# ═══════════════════════════════════════════════════════════════════════


class PlotlyRenderer:
    """Interactive Plotly visualizations for protein graphs."""

    @staticmethod
    def graph_3d(
        graph: ProteinGraph,
        color_by: str = "sequence_distance",
        title: str = "Protein Graph — 3-D",
        show_edges: bool = True,
        node_size: float = 6.0,
        edge_opacity: float = 0.5,
        height: int | None = None,
        camera_distance: float = 1.8,
    ) -> go.Figure:
        """3-D interactive protein graph visualization.

        Parameters
        ----------
        graph : ProteinGraph
        color_by : str
            'sequence_distance', 'contact_type', 'hydrophobicity',
            'charge', or 'residue_index'.
        title : str
        show_edges : bool
        node_size : float
        edge_opacity : float
        height : int | None
            Figure height in pixels. If None, uses Plotly default.
        camera_distance : float
            Camera eye multiplier. Higher = more zoomed out (default 1.8).

        Returns
        -------
        go.Figure
        """
        coords = graph.protein.ca_coordinates
        n = graph.n_nodes

        # ── Edge traces ──
        edge_traces = []
        if show_edges and graph.edge_data.n_edges > 0:
            ei = graph.edge_data.edge_index
            seq_dists = graph.edge_data.sequence_distances
            contact_types = classify_contacts(graph.edge_data)
            max_seq = int(seq_dists.max()) if len(seq_dists) > 0 else 1

            for e in range(graph.edge_data.n_edges):
                i, j = ei[0, e], ei[1, e]
                sd = int(seq_dists[e])
                ct = contact_types[e]

                if color_by == "sequence_distance":
                    color = _seq_dist_color(sd, max_seq)
                elif color_by == "contact_type":
                    color = _contact_color(ct)
                else:
                    color = "rgba(150,150,150,0.3)"

                edge_traces.append(go.Scatter3d(
                    x=[coords[i, 0], coords[j, 0], None],
                    y=[coords[i, 1], coords[j, 1], None],
                    z=[coords[i, 2], coords[j, 2], None],
                    mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="text",
                    text=f"{graph.protein.residues[i].name}{i+1} → "
                         f"{graph.protein.residues[j].name}{j+1} | "
                         f"d={graph.edge_data.distances[e]:.1f}Å | "
                         f"seq_dist={sd}",
                    showlegend=False,
                ))

        # ── Combine edges by contact type for legend ──
        # Group edge traces for compact rendering
        grouped_edge_traces = []
        if show_edges and graph.edge_data.n_edges > 0:
            contact_groups: Dict[str, List] = {
                "backbone": [], "short-range": [],
                "medium-range": [], "long-range": [],
            }
            ei = graph.edge_data.edge_index
            seq_dists = graph.edge_data.sequence_distances
            contact_types_list = classify_contacts(graph.edge_data)

            for e in range(graph.edge_data.n_edges):
                i, j = ei[0, e], ei[1, e]
                ct = contact_types_list[e]
                if ct in contact_groups:
                    contact_groups[ct].extend([
                        coords[i, 0], coords[j, 0], None,
                    ])

            # Build one trace per contact type
            for ct_name in ["backbone", "short-range", "medium-range", "long-range"]:
                xs = contact_groups.get(ct_name, [])
                if not xs:
                    continue

                n_pts = len(xs) // 3
                ys = []
                zs = []
                for e in range(graph.edge_data.n_edges):
                    if contact_types_list[e] == ct_name:
                        i, j = ei[0, e], ei[1, e]
                        ys.extend([coords[i, 1], coords[j, 1], None])
                        zs.extend([coords[i, 2], coords[j, 2], None])

                color = _contact_color(ct_name)
                grouped_edge_traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{ct_name} ({CONTACT_DESCRIPTIONS.get(ct_name, '')})",
                    hoverinfo="skip",
                    showlegend=True,
                ))

        # ── Node trace ──
        labels = [
            f"{r.name}{r.residue_number} ({r.three_letter})<br>"
            f"Hydro: {r.hydrophobicity:.1f} | Charge: {r.charge:.1f}<br>"
            f"Degree: {graph.adjacency.degree[i]}"
            for i, r in enumerate(graph.protein.residues)
        ]

        if color_by == "hydrophobicity":
            node_colors = [r.hydrophobicity for r in graph.protein.residues]
            colorscale = "RdBu_r"
            cbar_title = "Hydrophobicity"
        elif color_by == "charge":
            node_colors = [r.charge for r in graph.protein.residues]
            colorscale = "RdBu"
            cbar_title = "Charge"
        elif color_by == "residue_index":
            node_colors = list(range(n))
            colorscale = "Viridis"
            cbar_title = "Residue Index"
        else:
            node_colors = list(range(n))
            colorscale = "Viridis"
            cbar_title = "Residue Index"

        node_trace = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text",
            marker=dict(
                size=node_size,
                color=node_colors,
                colorscale=colorscale,
                colorbar=dict(title=cbar_title, thickness=15),
                line=dict(width=1, color="black"),
            ),
            text=[f"{r.name}{r.residue_number}" for r in graph.protein.residues],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=labels,
            hoverinfo="text",
            name="Residues",
            showlegend=True,
        )

        # ── Assemble figure ──
        fig = go.Figure(
            data=grouped_edge_traces + [node_trace],
        )

        # Compute a camera distance that fits the full protein
        _range = coords.max(axis=0) - coords.min(axis=0)
        _span = float(np.max(_range)) if np.max(_range) > 0 else 1.0
        _eye_dist = camera_distance  # multiplier – higher = more zoomed out
        _center = coords.mean(axis=0)

        layout_kwargs: dict = dict(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title="x (Å)",
                yaxis_title="y (Å)",
                zaxis_title="z (Å)",
                aspectmode="data",
                camera=dict(
                    eye=dict(
                        x=_eye_dist,
                        y=_eye_dist,
                        z=_eye_dist,
                    ),
                ),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(x=0.01, y=0.99),
            template="plotly_white",
        )
        if height is not None:
            layout_kwargs["height"] = height

        fig.update_layout(**layout_kwargs)

        return fig

    @staticmethod
    def adjacency_heatmap(
        graph: ProteinGraph,
        title: str = "Adjacency Matrix",
    ) -> go.Figure:
        """Heatmap of the adjacency matrix.

        Parameters
        ----------
        graph : ProteinGraph
        title : str

        Returns
        -------
        go.Figure
        """
        adj = graph.adjacency.sparse_matrix
        labels = [f"{r.name}{r.residue_number}"
                  for r in graph.protein.residues]

        fig = go.Figure(data=go.Heatmap(
            z=adj,
            x=labels,
            y=labels,
            colorscale=[[0, "#FFFFFF"], [1, "#1565C0"]],
            showscale=False,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Connected: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Residue j",
            yaxis_title="Residue i",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            width=700,
            height=700,
        )

        return fig

    @staticmethod
    def contact_map(
        graph: ProteinGraph,
        title: str = "Contact Map (colored by sequence distance)",
    ) -> go.Figure:
        """Contact map colored by sequence distance.

        Parameters
        ----------
        graph : ProteinGraph
        title : str

        Returns
        -------
        go.Figure
        """
        n = graph.n_nodes
        # Build color-coded matrix
        cmap = np.zeros((n, n))

        if graph.edge_data.n_edges > 0:
            ei = graph.edge_data.edge_index
            sd = graph.edge_data.sequence_distances
            for e in range(graph.edge_data.n_edges):
                i, j = ei[0, e], ei[1, e]
                cmap[i, j] = sd[e]
                cmap[j, i] = sd[e]

        labels = [f"{r.name}{r.residue_number}"
                  for r in graph.protein.residues]

        fig = go.Figure(data=go.Heatmap(
            z=cmap,
            x=labels,
            y=labels,
            colorscale="Hot_r",
            colorbar=dict(title="Sequence Distance"),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Seq Dist: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Residue j",
            yaxis_title="Residue i",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            width=700,
            height=700,
        )

        return fig

    @staticmethod
    def degree_histogram(
        analysis: FullGraphAnalysis,
        title: str = "Degree Distribution",
    ) -> go.Figure:
        """Histogram of node degrees.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        degrees = analysis.degree_distribution

        fig = go.Figure(data=go.Histogram(
            x=degrees,
            nbinsx=max(5, int(degrees.max() - degrees.min() + 1)),
            marker_color="#1565C0",
            opacity=0.8,
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Degree",
            yaxis_title="Count",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def distance_histogram(
        analysis: FullGraphAnalysis,
        title: str = "Edge Distance Distribution",
    ) -> go.Figure:
        """Histogram of edge distances.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        dists = analysis.graph.edge_data.distances

        fig = go.Figure(data=go.Histogram(
            x=dists,
            nbinsx=30,
            marker_color="#FF9800",
            opacity=0.8,
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Distance (Å)",
            yaxis_title="Count",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def sequence_distance_histogram(
        analysis: FullGraphAnalysis,
        title: str = "Sequence Distance Distribution",
    ) -> go.Figure:
        """Histogram of edge sequence distances, colored by contact type.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        seq_d = analysis.graph.edge_data.sequence_distances

        if len(seq_d) == 0:
            fig = go.Figure()
            fig.update_layout(title=dict(text=title, x=0.5))
            return fig

        # Split into contact types
        bb = seq_d[seq_d <= 1]
        sr = seq_d[(seq_d > 1) & (seq_d <= 4)]
        mr = seq_d[(seq_d > 4) & (seq_d <= 12)]
        lr = seq_d[seq_d > 12]

        fig = go.Figure()
        for data, name, color in [
            (bb, "Backbone", CONTACT_COLORS["backbone"]),
            (sr, "Short-range", CONTACT_COLORS["short-range"]),
            (mr, "Medium-range", CONTACT_COLORS["medium-range"]),
            (lr, "Long-range", CONTACT_COLORS["long-range"]),
        ]:
            if len(data) > 0:
                fig.add_trace(go.Histogram(
                    x=data,
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Sequence Distance |i − j|",
            yaxis_title="Count",
            barmode="stack",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def contact_type_pie(
        analysis: FullGraphAnalysis,
        title: str = "Contact Type Distribution",
    ) -> go.Figure:
        """Pie chart of contact types.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        labels = []
        values = []
        colors = []

        for ct in ["backbone", "short-range", "medium-range", "long-range"]:
            frac = analysis.contact_fractions.get(ct, 0.0)
            if frac > 0:
                labels.append(ct.capitalize())
                values.append(frac)
                colors.append(CONTACT_COLORS[ct])

        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.3,
            textinfo="label+percent",
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_white",
        )

        return fig

    @staticmethod
    def k_sweep_edges(
        sweep: KSweepAnalysis,
        title: str = "Edges vs. k",
    ) -> go.Figure:
        """Line plot of edge count vs. k.

        Parameters
        ----------
        sweep : KSweepAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sweep.k_values,
            y=sweep.edge_counts,
            mode="lines+markers",
            name="Total Edges",
            line=dict(color="#1565C0", width=3),
            marker=dict(size=8),
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="k (neighbors)",
            yaxis_title="Number of Edges",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def k_sweep_density(
        sweep: KSweepAnalysis,
        title: str = "Graph Density vs. k",
    ) -> go.Figure:
        """Line plot of density vs. k.

        Parameters
        ----------
        sweep : KSweepAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sweep.k_values,
            y=sweep.densities,
            mode="lines+markers",
            name="Density",
            line=dict(color="#FF9800", width=3),
            marker=dict(size=8),
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="k (neighbors)",
            yaxis_title="Graph Density",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def k_sweep_long_range(
        sweep: KSweepAnalysis,
        title: str = "Long-Range Contact Fraction vs. k",
    ) -> go.Figure:
        """Line plot of long-range fraction vs. k.

        Parameters
        ----------
        sweep : KSweepAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sweep.k_values,
            y=[f * 100 for f in sweep.long_range_fractions],
            mode="lines+markers",
            name="Long-Range %",
            line=dict(color="#F44336", width=3),
            marker=dict(size=8),
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="k (neighbors)",
            yaxis_title="Long-Range Contacts (%)",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def feature_heatmap(
        analysis: FeatureAnalysis,
        title: str = "Node Feature Matrix",
    ) -> go.Figure:
        """Heatmap of the node feature matrix.

        Parameters
        ----------
        analysis : FeatureAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        fm = analysis.feature_matrix
        res_labels = [
            f"{r.name}{r.residue_number}"
            for r in analysis.graph.protein.residues
        ]

        fig = go.Figure(data=go.Heatmap(
            z=fm,
            x=analysis.feature_names,
            y=res_labels,
            colorscale="Viridis",
            colorbar=dict(title="Value"),
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Feature",
            yaxis_title="Residue",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=max(400, fm.shape[0] * 12),
        )

        return fig

    @staticmethod
    def hydrophobicity_profile(
        analysis: FeatureAnalysis,
        title: str = "Hydrophobicity Profile",
    ) -> go.Figure:
        """Bar chart of per-residue hydrophobicity.

        Parameters
        ----------
        analysis : FeatureAnalysis
        title : str

        Returns
        -------
        go.Figure
        """
        residues = analysis.graph.protein.residues
        labels = [f"{r.name}{r.residue_number}" for r in residues]
        hydro = analysis.hydrophobicity_profile

        colors = ["#F44336" if h > 0 else "#2196F3" for h in hydro]

        fig = go.Figure(data=go.Bar(
            x=labels,
            y=hydro,
            marker_color=colors,
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Residue",
            yaxis_title="Hydrophobicity (Kyte-Doolittle)",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def preset_comparison_bars(
        comparison: PresetComparisonResult,
        title: str = "Preset Protein Comparison",
    ) -> go.Figure:
        """Grouped bar chart comparing all presets.

        Parameters
        ----------
        comparison : PresetComparisonResult
        title : str

        Returns
        -------
        go.Figure
        """
        names = [r["Protein"] for r in comparison.summary_table]
        edges = [r["Edges"] for r in comparison.summary_table]
        densities = [float(r["Density"]) * 1000 for r in comparison.summary_table]
        lr_pct = [float(r["Long-Range %"]) for r in comparison.summary_table]

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Edge Count", "Density (×10³)", "Long-Range %"])

        fig.add_trace(go.Bar(x=names, y=edges, marker_color="#1565C0", name="Edges"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=densities, marker_color="#FF9800", name="Density"), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=lr_pct, marker_color="#F44336", name="LR %"), row=1, col=3)

        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            template="plotly_white",
        )

        return fig


# ═══════════════════════════════════════════════════════════════════════
# MatplotlibRenderer
# ═══════════════════════════════════════════════════════════════════════


class MatplotlibRenderer:
    """Static Matplotlib figures for publication output."""

    @staticmethod
    def graph_2d(
        graph: ProteinGraph,
        title: str = "Protein Graph — 2-D Projection",
    ) -> Figure:
        """2-D projection of the protein graph (x-y plane).

        Parameters
        ----------
        graph : ProteinGraph
        title : str

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        coords = graph.protein.ca_coordinates

        # Draw edges colored by sequence distance
        if graph.edge_data.n_edges > 0:
            ei = graph.edge_data.edge_index
            seq_dists = graph.edge_data.sequence_distances
            max_sd = int(seq_dists.max()) if len(seq_dists) > 0 else 1
            contact_types = classify_contacts(graph.edge_data)

            for e in range(graph.edge_data.n_edges):
                i, j = ei[0, e], ei[1, e]
                ct = contact_types[e]
                color = _contact_color(ct)
                alpha = 0.3 if ct == "backbone" else 0.6
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=color, linewidth=1.0, alpha=alpha,
                )

        # Draw nodes
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=range(graph.n_nodes), cmap="viridis",
            s=60, edgecolors="black", linewidths=0.5, zorder=5,
        )

        # Label every 5th residue
        for i, r in enumerate(graph.protein.residues):
            if i % 5 == 0:
                ax.annotate(
                    f"{r.name}{r.residue_number}",
                    (coords[i, 0], coords[i, 1]),
                    fontsize=6, ha="center", va="bottom",
                )

        # Legend
        patches = [
            mpatches.Patch(color=CONTACT_COLORS[ct], label=ct.capitalize())
            for ct in ["backbone", "short-range", "medium-range", "long-range"]
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal")
        fig.tight_layout()

        return fig

    @staticmethod
    def adjacency_matrix(
        graph: ProteinGraph,
        title: str = "Adjacency Matrix",
    ) -> Figure:
        """Adjacency matrix as a heatmap.

        Parameters
        ----------
        graph : ProteinGraph
        title : str

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.imshow(graph.adjacency.sparse_matrix, cmap="Blues",
                  interpolation="nearest", aspect="equal")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Residue j")
        ax.set_ylabel("Residue i")
        fig.tight_layout()

        return fig

    @staticmethod
    def contact_type_bar(
        analysis: FullGraphAnalysis,
        title: str = "Contact Type Distribution",
    ) -> Figure:
        """Bar chart of contact types.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        types = ["backbone", "short-range", "medium-range", "long-range"]
        fracs = [analysis.contact_fractions.get(ct, 0.0) * 100 for ct in types]
        colors = [CONTACT_COLORS[ct] for ct in types]
        labels = [ct.capitalize() for ct in types]

        ax.bar(labels, fracs, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Fraction (%)")
        ax.set_title(title, fontsize=14)
        fig.tight_layout()

        return fig

    @staticmethod
    def degree_histogram(
        analysis: FullGraphAnalysis,
        title: str = "Degree Distribution",
    ) -> Figure:
        """Degree distribution histogram.

        Parameters
        ----------
        analysis : FullGraphAnalysis
        title : str

        Returns
        -------
        Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        degrees = analysis.degree_distribution
        ax.hist(degrees, bins=max(5, int(degrees.max() - degrees.min() + 1)),
                color="#1565C0", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=14)
        fig.tight_layout()

        return fig

    @staticmethod
    def k_sweep_summary(
        sweep: KSweepAnalysis,
        title: str = "k-Sweep Summary",
    ) -> Figure:
        """Multi-panel k-sweep summary.

        Parameters
        ----------
        sweep : KSweepAnalysis
        title : str

        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Edges vs k
        axes[0].plot(sweep.k_values, sweep.edge_counts,
                     "o-", color="#1565C0", linewidth=2)
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Edges")
        axes[0].set_title("Edges vs. k")

        # Density vs k
        axes[1].plot(sweep.k_values, sweep.densities,
                     "s-", color="#FF9800", linewidth=2)
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Density vs. k")

        # Long-range fraction vs k
        axes[2].plot(sweep.k_values,
                     [f * 100 for f in sweep.long_range_fractions],
                     "^-", color="#F44336", linewidth=2)
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("Long-Range (%)")
        axes[2].set_title("Long-Range Contacts vs. k")

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

        return fig

    @staticmethod
    def preset_comparison(
        comparison: PresetComparisonResult,
        title: str = "Preset Protein Comparison",
    ) -> Figure:
        """Grouped bar chart for preset comparison.

        Parameters
        ----------
        comparison : PresetComparisonResult
        title : str

        Returns
        -------
        Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        names = [r["Protein"] for r in comparison.summary_table]
        edges = [r["Edges"] for r in comparison.summary_table]
        densities = [float(r["Density"]) for r in comparison.summary_table]
        lr_pct = [float(r["Long-Range %"]) for r in comparison.summary_table]

        axes[0].bar(names, edges, color="#1565C0")
        axes[0].set_ylabel("Edges")
        axes[0].set_title("Edge Count")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(names, densities, color="#FF9800")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Graph Density")
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(names, lr_pct, color="#F44336")
        axes[2].set_ylabel("Long-Range %")
        axes[2].set_title("Long-Range Contacts")
        axes[2].tick_params(axis="x", rotation=45)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

        return fig
