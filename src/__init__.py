"""
The Graph Weaver — Protein-to-Graph Conversion.

Week 14, Project 3 — Biophysics Portfolio
Ryan Kamp | University of Cincinnati, Department of Computer Science

Converts protein structures into graph representations suitable for
Graph Neural Networks (GNNs). Implements k-NN edge construction via
KD-Tree, node featurization (one-hot, hydrophobicity, charge), edge
featurization (distance, direction, quaternions), sparse adjacency
matrices, and interactive visualization of the "neural view."

Modules
-------
graph_engine
    Core pipeline: PDB parsing, synthetic builders, node/edge features,
    k-NN construction, adjacency matrices, contact classification.
analysis
    High-level analysis pipelines and dataclasses for CLI/Streamlit.
visualization
    PlotlyRenderer (interactive 3-D) and MatplotlibRenderer (static).
"""

# ── Core engine ──
from src.graph_engine import (
    # Constants
    DEFAULT_K,
    DEFAULT_DISTANCE_CUTOFF,
    AMINO_ACIDS,
    AA_INDEX,
    NUM_AMINO_ACIDS,
    THREE_TO_ONE,
    ONE_TO_THREE,
    HYDROPHOBICITY,
    CHARGE,
    MOLECULAR_WEIGHT,
    HELIX_PROPENSITY,
    CONTACT_COLORS,
    CONTACT_DESCRIPTIONS,
    # Dataclasses
    Residue,
    ProteinStructure,
    NodeFeatures,
    EdgeData,
    AdjacencyData,
    ProteinGraph,
    GraphStatistics,
    # PDB parsing
    parse_pdb,
    parse_pdb_file,
    # Builders
    build_alpha_helix,
    build_beta_sheet,
    build_helix_turn_helix,
    build_beta_barrel,
    build_random_coil,
    build_two_domain,
    get_preset_proteins,
    # Pipeline functions
    compute_node_features,
    compute_edges,
    compute_adjacency,
    compute_graph_statistics,
    protein_to_graph,
    build_graph_from_pdb,
    sweep_k,
    # Contact helpers
    classify_contact,
    classify_contacts,
)

# ── Analysis pipelines ──
from src.analysis import (
    FullGraphAnalysis,
    KSweepAnalysis,
    ContactAnalysis,
    FeatureAnalysis,
    PresetComparisonResult,
    analyze_graph,
    analyze_k_sweep,
    analyze_contacts,
    analyze_features,
    compare_preset_proteins,
    graph_summary,
)

# ── Visualization ──
from src.visualization import (
    PlotlyRenderer,
    MatplotlibRenderer,
)
