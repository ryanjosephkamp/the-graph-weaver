"""
The Graph Weaver — Analysis Pipelines.

High-level analysis functions that wrap the core graph engine
to produce rich, annotated results for CLI and Streamlit output.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.graph_engine import (
    ProteinStructure,
    ProteinGraph,
    NodeFeatures,
    EdgeData,
    AdjacencyData,
    GraphStatistics,
    Residue,
    protein_to_graph,
    compute_node_features,
    compute_edges,
    compute_adjacency,
    compute_graph_statistics,
    classify_contacts,
    sweep_k,
    get_preset_proteins,
    AMINO_ACIDS,
    AA_INDEX,
    CONTACT_COLORS,
    CONTACT_DESCRIPTIONS,
    DEFAULT_K,
    DEFAULT_DISTANCE_CUTOFF,
)


# ═══════════════════════════════════════════════════════════════════════
# Analysis dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FullGraphAnalysis:
    """Complete analysis of a single protein graph.

    Attributes
    ----------
    protein : ProteinStructure
        Input protein.
    graph : ProteinGraph
        Constructed graph.
    statistics : GraphStatistics
        Summary statistics.
    contact_types : List[str]
        Contact classification for each edge.
    contact_fractions : Dict[str, float]
        Fraction of each contact type.
    aa_composition : Dict[str, int]
        Amino acid counts.
    aa_fractions : Dict[str, float]
        Amino acid fractions.
    degree_distribution : NDArray
        Histogram of node degrees.
    distance_histogram : Tuple[NDArray, NDArray]
        (counts, bin_edges) for edge distances.
    sequence_distance_histogram : Tuple[NDArray, NDArray]
        (counts, bin_edges) for sequence distances.
    explanation : str
        Human-readable summary.
    """
    protein: ProteinStructure
    graph: ProteinGraph
    statistics: GraphStatistics
    contact_types: List[str]
    contact_fractions: Dict[str, float]
    aa_composition: Dict[str, int]
    aa_fractions: Dict[str, float]
    degree_distribution: NDArray
    distance_histogram: Tuple[NDArray, NDArray]
    sequence_distance_histogram: Tuple[NDArray, NDArray]
    explanation: str


@dataclass
class KSweepAnalysis:
    """Analysis of graphs at multiple k values.

    Attributes
    ----------
    protein : ProteinStructure
        Input protein.
    k_values : List[int]
        k values tested.
    graphs : List[ProteinGraph]
        Graph at each k.
    statistics : List[GraphStatistics]
        Stats at each k.
    edge_counts : List[int]
        Number of edges at each k.
    densities : List[float]
        Graph density at each k.
    mean_degrees : List[float]
        Mean node degree at each k.
    long_range_fractions : List[float]
        Fraction of long-range contacts at each k.
    explanation : str
    """
    protein: ProteinStructure
    k_values: List[int]
    graphs: List[ProteinGraph]
    statistics: List[GraphStatistics]
    edge_counts: List[int]
    densities: List[float]
    mean_degrees: List[float]
    long_range_fractions: List[float]
    explanation: str


@dataclass
class ContactAnalysis:
    """Detailed contact-type analysis.

    Attributes
    ----------
    graph : ProteinGraph
        The protein graph.
    contact_types : List[str]
        Classification per edge.
    contact_counts : Dict[str, int]
        Count per contact type.
    contact_fractions : Dict[str, float]
        Fraction per contact type.
    backbone_edges : NDArray
        (2, E_bb) edge indices for backbone contacts.
    short_range_edges : NDArray
        (2, E_sr) edge indices for short-range contacts.
    medium_range_edges : NDArray
        (2, E_mr) edge indices for medium-range contacts.
    long_range_edges : NDArray
        (2, E_lr) edge indices for long-range contacts.
    explanation : str
    """
    graph: ProteinGraph
    contact_types: List[str]
    contact_counts: Dict[str, int]
    contact_fractions: Dict[str, float]
    backbone_edges: NDArray
    short_range_edges: NDArray
    medium_range_edges: NDArray
    long_range_edges: NDArray
    explanation: str


@dataclass
class FeatureAnalysis:
    """Detailed node feature analysis.

    Attributes
    ----------
    graph : ProteinGraph
        The protein graph.
    feature_matrix : NDArray
        (N, D) node features.
    feature_names : List[str]
        Column names.
    feature_means : NDArray
        (D,) mean per feature.
    feature_stds : NDArray
        (D,) std per feature.
    hydrophobicity_profile : NDArray
        (N,) raw hydrophobicity values.
    charge_profile : NDArray
        (N,) raw charge values.
    explanation : str
    """
    graph: ProteinGraph
    feature_matrix: NDArray
    feature_names: List[str]
    feature_means: NDArray
    feature_stds: NDArray
    hydrophobicity_profile: NDArray
    charge_profile: NDArray
    explanation: str


@dataclass
class PresetComparisonResult:
    """Comparison of all preset proteins.

    Attributes
    ----------
    analyses : Dict[str, FullGraphAnalysis]
        Analysis per preset.
    summary_table : List[Dict]
        Tabular summary for display.
    """
    analyses: Dict[str, FullGraphAnalysis]
    summary_table: List[Dict]


# ═══════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════


def analyze_graph(
    protein: ProteinStructure,
    k: int = DEFAULT_K,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> FullGraphAnalysis:
    """Full analysis of a single protein graph.

    Parameters
    ----------
    protein : ProteinStructure
        Input protein.
    k : int
        k-NN parameter.
    distance_cutoff : float
        Edge distance cutoff (Å).

    Returns
    -------
    FullGraphAnalysis
    """
    graph = protein_to_graph(protein, k=k, distance_cutoff=distance_cutoff)
    stats = compute_graph_statistics(graph)
    contact_types = classify_contacts(graph.edge_data)

    # Contact fractions
    contact_counts: Dict[str, int] = {}
    for ct in contact_types:
        contact_counts[ct] = contact_counts.get(ct, 0) + 1
    total_edges = max(len(contact_types), 1)
    contact_fractions = {k_: v / total_edges for k_, v in contact_counts.items()}

    # AA composition
    aa_comp: Dict[str, int] = {}
    for res in protein.residues:
        aa_comp[res.name] = aa_comp.get(res.name, 0) + 1
    aa_frac = {k_: v / protein.n_residues for k_, v in aa_comp.items()}

    # Degree distribution
    degree_dist = graph.adjacency.degree

    # Distance histogram
    if graph.edge_data.n_edges > 0:
        d_hist = np.histogram(graph.edge_data.distances, bins=30)
        s_hist = np.histogram(graph.edge_data.sequence_distances,
                              bins=min(30, max(1, int(graph.edge_data.sequence_distances.max()))))
    else:
        d_hist = (np.array([0]), np.array([0, 1]))
        s_hist = (np.array([0]), np.array([0, 1]))

    explanation = textwrap.dedent(f"""\
        Graph analysis for "{protein.name}":
        • {stats.n_nodes} nodes (residues), {stats.n_edges} edges
        • k = {k}, cutoff = {distance_cutoff:.1f} Å
        • Mean degree: {stats.mean_degree:.1f} ± {stats.std_degree:.1f}
        • Density: {stats.density:.4f}
        • Short-range edges (≤4): {stats.n_short_range} ({stats.n_short_range/max(total_edges,1)*100:.1f}%)
        • Medium-range edges (5–12): {stats.n_medium_range} ({stats.n_medium_range/max(total_edges,1)*100:.1f}%)
        • Long-range edges (>12): {stats.n_long_range} ({stats.n_long_range/max(total_edges,1)*100:.1f}%)
        • Feature dimension: {stats.feature_dim}
    """)

    return FullGraphAnalysis(
        protein=protein,
        graph=graph,
        statistics=stats,
        contact_types=contact_types,
        contact_fractions=contact_fractions,
        aa_composition=aa_comp,
        aa_fractions=aa_frac,
        degree_distribution=degree_dist,
        distance_histogram=d_hist,
        sequence_distance_histogram=s_hist,
        explanation=explanation,
    )


def analyze_k_sweep(
    protein: ProteinStructure,
    k_values: Optional[List[int]] = None,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> KSweepAnalysis:
    """Sweep k and analyse how the graph changes.

    Parameters
    ----------
    protein : ProteinStructure
        Input protein.
    k_values : List[int], optional
        k values to test.
    distance_cutoff : float
        Edge distance cutoff (Å).

    Returns
    -------
    KSweepAnalysis
    """
    if k_values is None:
        k_values = [2, 4, 6, 8, 10, 15, 20]

    graphs = sweep_k(protein, k_values=k_values, distance_cutoff=distance_cutoff)
    all_stats = [compute_graph_statistics(g) for g in graphs]

    edge_counts = [s.n_edges for s in all_stats]
    densities = [s.density for s in all_stats]
    mean_degrees = [s.mean_degree for s in all_stats]

    long_range_fracs = []
    for g in graphs:
        seq_d = g.edge_data.sequence_distances
        if len(seq_d) > 0:
            lr = float(np.sum(seq_d > 12)) / len(seq_d)
        else:
            lr = 0.0
        long_range_fracs.append(lr)

    explanation = textwrap.dedent(f"""\
        k-sweep for "{protein.name}" ({protein.n_residues} residues):
        • k values tested: {k_values}
        • Edge counts range: {min(edge_counts)} → {max(edge_counts)}
        • Density range: {min(densities):.4f} → {max(densities):.4f}
        • At k={k_values[0]}: only local backbone links
        • At k={k_values[-1]}: dense graph capturing tertiary contacts
        • Long-range fraction increases with k, revealing folding information.
    """)

    return KSweepAnalysis(
        protein=protein,
        k_values=k_values,
        graphs=graphs,
        statistics=all_stats,
        edge_counts=edge_counts,
        densities=densities,
        mean_degrees=mean_degrees,
        long_range_fractions=long_range_fracs,
        explanation=explanation,
    )


def analyze_contacts(
    graph: ProteinGraph,
) -> ContactAnalysis:
    """Detailed contact-type analysis.

    Parameters
    ----------
    graph : ProteinGraph
        The protein graph.

    Returns
    -------
    ContactAnalysis
    """
    contact_types = classify_contacts(graph.edge_data)

    contact_counts: Dict[str, int] = {
        "backbone": 0, "short-range": 0,
        "medium-range": 0, "long-range": 0,
    }
    for ct in contact_types:
        contact_counts[ct] = contact_counts.get(ct, 0) + 1

    total = max(len(contact_types), 1)
    contact_fractions = {k_: v / total for k_, v in contact_counts.items()}

    # Extract edge subsets
    edge_idx = graph.edge_data.edge_index
    seq_d = graph.edge_data.sequence_distances

    def _extract(mask: NDArray) -> NDArray:
        if mask.sum() == 0:
            return np.zeros((2, 0), dtype=np.int64)
        return edge_idx[:, mask]

    bb_mask = seq_d <= 1
    sr_mask = (seq_d > 1) & (seq_d <= 4)
    mr_mask = (seq_d > 4) & (seq_d <= 12)
    lr_mask = seq_d > 12

    explanation = textwrap.dedent(f"""\
        Contact analysis for "{graph.name}":
        • Backbone (i±1): {contact_counts['backbone']} ({contact_fractions['backbone']*100:.1f}%)
        • Short-range (i±2 to i±4): {contact_counts['short-range']} ({contact_fractions['short-range']*100:.1f}%)
          → α-helix hydrogen bonds (i to i+4)
        • Medium-range (i±5 to i±12): {contact_counts['medium-range']} ({contact_fractions['medium-range']*100:.1f}%)
          → Turns, loops, and β-hairpin contacts
        • Long-range (i±13+): {contact_counts['long-range']} ({contact_fractions['long-range']*100:.1f}%)
          → Tertiary contacts — the hardest part of structure prediction
    """)

    return ContactAnalysis(
        graph=graph,
        contact_types=contact_types,
        contact_counts=contact_counts,
        contact_fractions=contact_fractions,
        backbone_edges=_extract(bb_mask),
        short_range_edges=_extract(sr_mask),
        medium_range_edges=_extract(mr_mask),
        long_range_edges=_extract(lr_mask),
        explanation=explanation,
    )


def analyze_features(
    graph: ProteinGraph,
) -> FeatureAnalysis:
    """Detailed node feature analysis.

    Parameters
    ----------
    graph : ProteinGraph
        The protein graph.

    Returns
    -------
    FeatureAnalysis
    """
    fm = graph.node_features.feature_matrix
    means = fm.mean(axis=0)
    stds = fm.std(axis=0)

    explanation = textwrap.dedent(f"""\
        Feature analysis for "{graph.name}":
        • Feature dimension: {fm.shape[1]}
        • Features: {', '.join(graph.node_features.feature_names[:5])}...
        • One-hot encoding: {fm.shape[1] - 4} amino acid channels
        • Scalar features: hydrophobicity, charge, weight, helix_propensity
        • Mean hydrophobicity: {graph.node_features.hydrophobicity.mean():.2f}
        • Mean charge: {graph.node_features.charge.mean():.2f}
    """)

    return FeatureAnalysis(
        graph=graph,
        feature_matrix=fm,
        feature_names=graph.node_features.feature_names,
        feature_means=means,
        feature_stds=stds,
        hydrophobicity_profile=graph.node_features.hydrophobicity,
        charge_profile=graph.node_features.charge,
        explanation=explanation,
    )


def compare_preset_proteins(
    k: int = DEFAULT_K,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> PresetComparisonResult:
    """Analyse all six preset proteins and compare.

    Parameters
    ----------
    k : int
        k-NN parameter.
    distance_cutoff : float
        Edge cutoff (Å).

    Returns
    -------
    PresetComparisonResult
    """
    presets = get_preset_proteins()
    analyses: Dict[str, FullGraphAnalysis] = {}

    for name, protein in presets.items():
        analyses[name] = analyze_graph(protein, k=k, distance_cutoff=distance_cutoff)

    summary_table = []
    for name, analysis in analyses.items():
        s = analysis.statistics
        summary_table.append({
            "Protein": name,
            "Residues": s.n_nodes,
            "Edges": s.n_edges,
            "Density": f"{s.density:.4f}",
            "Mean Degree": f"{s.mean_degree:.1f}",
            "Short-Range %": f"{s.n_short_range / max(s.n_edges, 1) * 100:.1f}",
            "Medium-Range %": f"{s.n_medium_range / max(s.n_edges, 1) * 100:.1f}",
            "Long-Range %": f"{s.n_long_range / max(s.n_edges, 1) * 100:.1f}",
            "Feature Dim": s.feature_dim,
        })

    return PresetComparisonResult(
        analyses=analyses,
        summary_table=summary_table,
    )


def graph_summary(analysis: FullGraphAnalysis) -> str:
    """Human-readable text summary of a graph analysis.

    Parameters
    ----------
    analysis : FullGraphAnalysis
        The analysis result.

    Returns
    -------
    str
        Multi-line summary string.
    """
    s = analysis.statistics
    lines = [
        f"═══ Protein Graph Summary: {analysis.protein.name} ═══",
        "",
        f"  Residues:          {s.n_nodes}",
        f"  Sequence:          {analysis.protein.sequence[:40]}{'...' if len(analysis.protein.sequence) > 40 else ''}",
        f"  Edges:             {s.n_edges}",
        f"  k:                 {s.k}",
        f"  Distance cutoff:   {s.distance_cutoff:.1f} Å",
        f"  Density:           {s.density:.4f}",
        "",
        f"  Mean degree:       {s.mean_degree:.1f} ± {s.std_degree:.1f}",
        f"  Degree range:      [{s.min_degree}, {s.max_degree}]",
        "",
        f"  Mean edge dist:    {s.mean_edge_distance:.2f} ± {s.std_edge_distance:.2f} Å",
        f"  Mean seq dist:     {s.mean_sequence_distance:.1f} ± {s.std_sequence_distance:.1f}",
        "",
        "  Contact breakdown:",
    ]

    for ctype in ["backbone", "short-range", "medium-range", "long-range"]:
        frac = analysis.contact_fractions.get(ctype, 0.0)
        desc = CONTACT_DESCRIPTIONS.get(ctype, "")
        lines.append(f"    {ctype:15s}: {frac*100:5.1f}%  — {desc}")

    lines.extend([
        "",
        f"  Feature dimension: {s.feature_dim}",
        "",
        "  AA composition (top 5):",
    ])

    sorted_aa = sorted(analysis.aa_composition.items(),
                        key=lambda x: x[1], reverse=True)[:5]
    for aa, count in sorted_aa:
        lines.append(f"    {aa}: {count} ({count / s.n_nodes * 100:.1f}%)")

    return "\n".join(lines)
