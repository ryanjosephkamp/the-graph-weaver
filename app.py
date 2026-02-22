"""
The Graph Weaver — Streamlit Application.

Six-page interactive app exploring protein-to-graph conversion,
k-NN edge construction, node/edge featurization, contact-type
analysis, the "k Slider" interactive demo, and GNN theory:

    1. Home & Overview
    2. The Neural View (interactive 3-D graph + edge coloring)
    3. The k Slider (interactive k sweep)
    4. The Contact Map (contact-type analysis)
    5. Protein Comparison (all presets side by side)
    6. Theory & Mathematics
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine import (
    ProteinStructure,
    ProteinGraph,
    build_alpha_helix,
    build_beta_sheet,
    build_helix_turn_helix,
    build_beta_barrel,
    build_random_coil,
    build_two_domain,
    get_preset_proteins,
    parse_pdb,
    protein_to_graph,
    compute_graph_statistics,
    classify_contacts,
    AMINO_ACIDS,
    HYDROPHOBICITY,
    CHARGE,
    CONTACT_COLORS,
    CONTACT_DESCRIPTIONS,
    DEFAULT_K,
    DEFAULT_DISTANCE_CUTOFF,
)
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
from src.visualization import (
    PlotlyRenderer,
    MatplotlibRenderer,
)


# ═══════════════════════════════════════════════════════════════════════
# Uploaded PDB helpers
# ═══════════════════════════════════════════════════════════════════════

_UPLOADED_PROTEIN_LABEL = "📄 Uploaded PDB"


def _get_uploaded_protein() -> Optional[ProteinStructure]:
    """Return the uploaded protein from session_state, or None."""
    return st.session_state.get("uploaded_protein", None)


def _resolve_protein(protein_name: str) -> ProteinStructure:
    """Return a ProteinStructure for *protein_name*.

    If the name matches the uploaded-PDB sentinel, pull it from
    session_state; otherwise look it up in the preset catalogue.
    """
    if protein_name == _UPLOADED_PROTEIN_LABEL:
        uploaded = _get_uploaded_protein()
        if uploaded is None:
            raise ValueError("No uploaded PDB in session state.")
        return uploaded
    return get_preset_proteins()[protein_name]


# ═══════════════════════════════════════════════════════════════════════
# Cached helpers
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner=False)
def _cached_analysis(
    protein_name: str, k: int, cutoff: float,
    _protein_override: Optional[ProteinStructure] = None,
) -> FullGraphAnalysis:
    if _protein_override is not None:
        return analyze_graph(_protein_override, k=k, distance_cutoff=cutoff)
    proteins = get_preset_proteins()
    protein = proteins[protein_name]
    return analyze_graph(protein, k=k, distance_cutoff=cutoff)


@st.cache_data(show_spinner=False)
def _cached_k_sweep(
    protein_name: str, cutoff: float,
    _protein_override: Optional[ProteinStructure] = None,
) -> KSweepAnalysis:
    if _protein_override is not None:
        return analyze_k_sweep(_protein_override, distance_cutoff=cutoff)
    proteins = get_preset_proteins()
    protein = proteins[protein_name]
    return analyze_k_sweep(protein, distance_cutoff=cutoff)


@st.cache_data(show_spinner=False)
def _cached_preset_comparison(
    k: int, cutoff: float,
) -> PresetComparisonResult:
    return compare_preset_proteins(k=k, distance_cutoff=cutoff)


def _get_analysis(protein_name: str, k: int, cutoff: float) -> FullGraphAnalysis:
    """Resolve *protein_name* (preset or uploaded) and return the analysis."""
    if protein_name == _UPLOADED_PROTEIN_LABEL:
        protein = _get_uploaded_protein()
        if protein is None:
            st.error("No PDB file uploaded. Please upload a .pdb file in the sidebar.")
            st.stop()
        # Use the PDB content hash as a cache-buster so different files
        # with the same name don't collide.
        pdb_hash = st.session_state.get("uploaded_pdb_hash", 0)
        cache_key = f"__uploaded__{pdb_hash}"
        return _cached_analysis(cache_key, k, cutoff, _protein_override=protein)
    return _cached_analysis(protein_name, k, cutoff)


def _get_k_sweep(protein_name: str, cutoff: float) -> KSweepAnalysis:
    """Resolve *protein_name* (preset or uploaded) and return the k-sweep."""
    if protein_name == _UPLOADED_PROTEIN_LABEL:
        protein = _get_uploaded_protein()
        if protein is None:
            st.error("No PDB file uploaded. Please upload a .pdb file in the sidebar.")
            st.stop()
        pdb_hash = st.session_state.get("uploaded_pdb_hash", 0)
        cache_key = f"__uploaded__{pdb_hash}"
        return _cached_k_sweep(cache_key, cutoff, _protein_override=protein)
    return _cached_k_sweep(protein_name, cutoff)


# ═══════════════════════════════════════════════════════════════════════
# App configuration
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Graph Weaver",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "🏠 Home",
    "🧠 The Neural View",
    "🎚️ The k Slider",
    "📋 The Contact Map",
    "📊 Protein Comparison",
    "📚 Theory & Mathematics",
]

FOOTER = """
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <b>The Graph Weaver</b> – Week 14 Project 3 | Biophysics Portfolio<br>
    Ryan Kamp | University of Cincinnati Department of Computer Science<br>
    <a href="mailto:kamprj@mail.uc.edu">kamprj@mail.uc.edu</a> |
    <a href="https://github.com/ryanjosephkamp/the-graph-weaver">GitHub</a><br>
    February 22, 2026
</div>
"""

_PROTEIN_MAP: Dict[str, callable] = {
    "α-Helix": lambda: build_alpha_helix(),
    "β-Sheet": lambda: build_beta_sheet(),
    "Helix-Turn-Helix": lambda: build_helix_turn_helix(),
    "β-Barrel": lambda: build_beta_barrel(),
    "Random Coil": lambda: build_random_coil(),
    "Two-Domain Protein": lambda: build_two_domain(),
}

_PRESET_OPTIONS = list(_PROTEIN_MAP.keys())


def _protein_options() -> list:
    """Return the list of available protein names.

    Includes the uploaded PDB entry when a file has been uploaded.
    """
    opts = list(_PRESET_OPTIONS)
    if _get_uploaded_protein() is not None:
        opts.insert(0, _UPLOADED_PROTEIN_LABEL)
    return opts


# ═══════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════


def _render_pdb_upload() -> None:
    """Render the PDB file uploader in the sidebar."""
    st.sidebar.markdown("### Upload a PDB File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload .pdb",
        type=["pdb"],
        help="Upload a PDB file to analyse your own protein structure. "
             "The Cα atoms will be extracted automatically.",
    )

    if uploaded_file is not None:
        pdb_text = uploaded_file.getvalue().decode("utf-8")
        file_name = uploaded_file.name
        protein_name = os.path.splitext(file_name)[0]

        # Only re-parse when the file content changes
        current_hash = hash(pdb_text)
        if (
            st.session_state.get("uploaded_pdb_hash") != current_hash
        ):
            protein = parse_pdb(pdb_text, name=protein_name)
            if protein.n_residues == 0:
                st.sidebar.error(
                    "No Cα atoms found in the uploaded PDB file. "
                    "Please upload a valid PDB with ATOM records."
                )
                return
            st.session_state["uploaded_protein"] = protein
            st.session_state["uploaded_pdb_hash"] = current_hash
            st.session_state["uploaded_pdb_name"] = protein_name

        protein = st.session_state.get("uploaded_protein")
        if protein is not None:
            st.sidebar.success(
                f"✅ **{protein.name}** — {protein.n_residues} residues loaded"
            )
    else:
        # File uploader was cleared — remove uploaded protein
        if "uploaded_protein" in st.session_state:
            del st.session_state["uploaded_protein"]
        if "uploaded_pdb_hash" in st.session_state:
            del st.session_state["uploaded_pdb_hash"]
        if "uploaded_pdb_name" in st.session_state:
            del st.session_state["uploaded_pdb_name"]


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    st.sidebar.title("🕸️ The Graph Weaver")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", PAGES, index=0)
    st.sidebar.markdown("---")

    # PDB upload widget (available on every page)
    _render_pdb_upload()
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "*Week 14, Project 3 — Biophysics Portfolio*\n\n"
        "*Ryan Kamp • University of Cincinnati*"
    )
    return page


def _init_shared_state() -> None:
    """Ensure shared sidebar keys exist in session_state."""
    defaults = {
        "shared_protein": "α-Helix",
        "shared_k": DEFAULT_K,
        "shared_cutoff": DEFAULT_DISTANCE_CUTOFF,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _graph_sidebar() -> dict:
    """Sidebar controls for graph parameters."""
    _init_shared_state()

    st.sidebar.markdown("### Graph Parameters")

    protein = st.sidebar.selectbox(
        "Protein Structure",
        _protein_options(),
        key="shared_protein",
        help="Choose a preset protein structure, or select your uploaded PDB.",
    )

    k = st.sidebar.slider(
        "k (neighbors)",
        2, 30, step=1,
        key="shared_k",
        help="Number of k-nearest neighbors for edge construction.",
    )

    cutoff = st.sidebar.slider(
        "Distance cutoff (Å)",
        5.0, 30.0, step=0.5,
        key="shared_cutoff",
        help="Maximum distance for edge construction.",
    )

    return {"protein": protein, "k": k, "cutoff": cutoff}


# ═══════════════════════════════════════════════════════════════════════
# Page 1 — Home
# ═══════════════════════════════════════════════════════════════════════


def page_home() -> None:
    """Render the landing page."""
    st.title("🕸️ The Graph Weaver")
    st.subheader(
        "Protein-to-Graph Conversion — "
        "From Atomic Coordinates to Learnable Features"
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
**The Graph Weaver** implements a complete pipeline for converting
protein structures into graph representations suitable for
**Graph Neural Networks (GNNs)**. This is the direct bridge between
structural biology and geometric deep learning.

**How do you feed a protein into PyTorch?**
You don't use a CNN — proteins aren't square images. You use a **Graph**.

This project builds a pipeline that converts a `.pdb` file into a
`geometric_data` object with:

- **Nodes:** Each amino acid (Cα atom)
- **Node Features:** One-hot encoding, hydrophobicity, charge, molecular weight, helix propensity
- **Edges:** k-Nearest Neighbors within a distance cutoff (KD-Tree)
- **Edge Features:** Euclidean distance, directional vector, sequence distance, orientation quaternion
- **Output:** Sparse Adjacency Matrix and Feature Matrix
        """)

    with col2:
        st.markdown("""
#### Key Equations

**k-NN Edge Construction:**

$$d_{ij} = ||\\mathbf{r}_i - \\mathbf{r}_j||_2$$

**Direction Vector:**

$$\\hat{\\mathbf{r}}_{ij} = \\frac{\\mathbf{r}_j - \\mathbf{r}_i}{d_{ij}}$$

**Adjacency Matrix:**

$$A_{ij} = \\begin{cases} 1 & \\text{if } j \\in \\text{kNN}(i) \\\\ 0 & \\text{otherwise} \\end{cases}$$
        """)

    st.markdown("---")

    # Quick preview — use uploaded protein if available, else α-Helix
    uploaded = _get_uploaded_protein()
    if uploaded is not None:
        preview_name = _UPLOADED_PROTEIN_LABEL
    else:
        preview_name = "α-Helix"
    with st.spinner("Building graph preview..."):
        analysis = _get_analysis(preview_name, DEFAULT_K, DEFAULT_DISTANCE_CUTOFF)

    preview_title = f"Preview — {analysis.protein.name} Protein Graph (k={DEFAULT_K})"
    fig_preview = PlotlyRenderer.graph_3d(
        analysis.graph,
        color_by="contact_type",
        title=preview_title,
        height=750,
        camera_distance=3.5,
    )
    st.plotly_chart(fig_preview, use_container_width=True)

    with st.expander("ℹ️ About This Preview"):
        st.markdown("""
This shows an α-helix backbone converted into a graph:
- **Nodes** (colored dots) are individual amino acid residues at their Cα positions
- **Edges** (lines) connect residues that are spatial neighbors (k=10 nearest)
- Edge colors encode **contact type**:
    - **Grey:** backbone (i±1)
    - **Blue:** short-range (i±2 to i±4) — α-helix hydrogen bonds
    - **Orange:** medium-range (i±5 to i±12) — turns and loops
    - **Red:** long-range (i±13+) — tertiary contacts
- This is exactly what a GNN "sees" when it processes a protein structure.
        """)

    st.markdown("---")

    st.markdown("### Quick Start")
    st.markdown("""
| Page | What You'll Find |
|------|------------------|
| 🧠 **The Neural View** | Interactive 3-D graph with edges colored by sequence distance, contact type, or residue property |
| 🎚️ **The k Slider** | Slide k from 2 to 20 and watch the graph evolve — local backbone → dense tertiary contacts |
| 📋 **The Contact Map** | Contact map, adjacency matrix, contact-type breakdown, node feature heatmap |
| 📊 **Protein Comparison** | All six preset proteins compared side by side |
| 📚 **Theory & Mathematics** | Full derivations: k-NN graphs, GNN message passing, feature engineering, quaternion edges |
    """)

    with st.expander("ℹ️ About the Quick Start Table"):
        st.markdown("""
Each page explores a different aspect of the protein-to-graph pipeline:

- **The Neural View** is the core visualization — it shows exactly what a GNN
  "sees" when it processes a protein. The edges are the information channels.
- **The k Slider** is the interactive demo. At k=4, only local backbone links
  are present. At k=20, the graph captures all the folding information.
- **The Contact Map** provides the analytical view — adjacency matrices, contact
  distributions, and the node feature matrix.
- **Protein Comparison** shows how different structural motifs (helix, sheet,
  barrel, coil) produce fundamentally different graph topologies.
- **Theory & Mathematics** contains the full mathematical framework for
  protein graph construction and GNN featurization.
        """)

    st.markdown("---")

    st.markdown("### The Science Behind It")

    with st.expander("Why Graphs for Proteins?"):
        st.markdown("""
Proteins are **irregular 3-D structures** — they don't sit on a grid.
CNNs work on regular grids (images), but proteins need a representation
that respects their **geometry**.

A graph is the natural choice:
- Each residue becomes a **node** with biochemical features
- Spatial proximity becomes **edges** with geometric features
- The **adjacency matrix** encodes the 3-D topology
- **GNN message passing** aggregates information from neighbors

This is how state-of-the-art models like AlphaFold2, ESMFold, and
RoseTTAFold process protein structures internally.
        """)

    with st.expander("What is k-NN Graph Construction?"):
        st.markdown("""
**k-Nearest Neighbors (k-NN)** is the algorithm that decides which
residues are "neighbors" in the graph:

1. Place all Cα atoms in a **KD-Tree** for O(N log N) spatial queries
2. For each residue i, find its **k nearest** Cα atoms
3. Add an edge (i → j) if the distance is within the **cutoff** (10Å default)
4. Compute **edge features**: distance, direction vector, quaternion

The parameter **k** controls the graph density:
- **k = 2–4:** Only backbone neighbors — very sparse
- **k = 8–12:** Local + medium-range contacts — good for secondary structure
- **k = 15–20:** Captures long-range tertiary contacts — reveals folding
        """)

    with st.expander("What Are Sequence Distance Contacts?"):
        st.markdown("""
**Sequence distance** |i − j| measures how far apart two residues are
in the amino acid chain:

| Sequence Distance | Type | Structural Meaning |
|---|---|---|
| |i−j| = 1 | Backbone | Sequential bond |
| |i−j| = 2–4 | Short-range | α-helix hydrogen bonds (i to i+4) |
| |i−j| = 5–12 | Medium-range | Turns, loops, β-hairpins |
| |i−j| > 12 | Long-range | **Tertiary contacts** — the hard part of folding |

Long-range contacts are the most informative for structure prediction.
They tell us that residues far apart in the sequence are close in 3-D
space — this is exactly what makes protein folding a hard problem.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 2 — The Neural View
# ═══════════════════════════════════════════════════════════════════════


def page_neural_view() -> None:
    """Interactive 3-D protein graph visualization."""
    st.title("🧠 The Neural View")
    st.markdown(
        "Explore the protein graph in 3-D. Each node is a residue (Cα atom), "
        "and edges connect spatial neighbors. **This is exactly what a GNN sees** "
        "when it processes a protein structure."
    )
    st.markdown("---")

    params = _graph_sidebar()

    with st.spinner("Building protein graph..."):
        analysis = _get_analysis(params["protein"], params["k"], params["cutoff"])

    graph = analysis.graph
    stats = analysis.statistics

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", f"{stats.n_nodes}")
    col2.metric("Edges", f"{stats.n_edges:,}")
    col3.metric("Mean Degree", f"{stats.mean_degree:.1f}")
    col4.metric("Density", f"{stats.density:.4f}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Nodes:** Number of amino acid residues (Cα atoms) in the protein.
- **Edges:** Number of connections in the graph. Each edge represents
  a spatial proximity relationship between two residues.
- **Mean Degree:** Average number of neighbors per node. Higher degree
  means more information flow in the GNN.
- **Density:** Fraction of all possible edges that are present.
  Density = E / (N × (N−1)). Protein graphs are typically sparse.
        """)

    st.markdown("---")

    # ── Color mode selector ──
    color_mode = st.selectbox(
        "Color edges/nodes by:",
        ["Contact Type (sequence distance)", "Hydrophobicity", "Charge", "Residue Index"],
        help="Choose how to color the graph edges and nodes.",
    )

    color_map = {
        "Contact Type (sequence distance)": "contact_type",
        "Hydrophobicity": "hydrophobicity",
        "Charge": "charge",
        "Residue Index": "residue_index",
    }
    color_by = color_map[color_mode]

    # ── 3-D Graph ──
    st.markdown("### 3-D Protein Graph")
    fig_3d = PlotlyRenderer.graph_3d(
        graph,
        color_by=color_by,
        title=f"Protein Graph — {params['protein']} (k={params['k']})",
        height=750,
        camera_distance=3.5,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    with st.expander("ℹ️ Reading the 3-D Graph"):
        st.markdown("""
**The Neural View** shows the protein as a GNN would process it:

- **Nodes** are amino acid residues at their Cα coordinates
- **Edges** connect spatially proximal residues (k-NN + distance cutoff)
- **Edge colors** (in Contact Type mode):
    - **Grey:** backbone (i±1) — the peptide chain itself
    - **Blue:** short-range (i±2 to i±4) — α-helix contacts
    - **Orange:** medium-range (i±5 to i±12) — turns and loops
    - **Red:** long-range (i±13+) — tertiary contacts, the hardest to predict

**Interaction:** Rotate, zoom, and hover to inspect individual residues
and edges. The hover text shows amino acid identity, hydrophobicity,
charge, and node degree.

**The Twist:** Notice that short-range edges (blue) dominate in
α-helices, while long-range edges (red) appear in barrel and
two-domain structures. This visualizes the fundamental difference
between secondary and tertiary structure.
        """)

    st.markdown("---")

    # ── Contact breakdown ──
    st.markdown("### Contact-Type Breakdown")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_pie = PlotlyRenderer.contact_type_pie(
            analysis,
            title=f"Contact Types — {params['protein']}",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_seq = PlotlyRenderer.sequence_distance_histogram(
            analysis,
            title=f"Sequence Distance Distribution — {params['protein']}",
        )
        st.plotly_chart(fig_seq, use_container_width=True)

    with st.expander("ℹ️ Interpreting the Contact Breakdown"):
        st.markdown("""
**Contact Types** classify edges by how far apart the connected
residues are in the amino acid sequence:

- **Backbone (grey):** i±1 — the chemical bond along the polypeptide chain.
- **Short-range (blue):** i±2 to i±4 — these are the hydrogen bonds that
  form α-helices (the i→i+4 rule).
- **Medium-range (orange):** i±5 to i±12 — turns, loops, and β-hairpin contacts.
- **Long-range (red):** i±13+ — these are **tertiary contacts** that define
  the 3-D fold. They are the hardest to predict and the most informative for GNNs.

The **sequence distance histogram** shows the full distribution. For α-helices,
you'll see a peak at |i−j| = 3–4 (helical periodicity). For β-sheets, you'll
see peaks at the strand separation distance.
        """)

    st.markdown("---")

    # ── Edge distance distribution ──
    st.markdown("### Edge Distance Distribution")
    col_c, col_d = st.columns(2)

    with col_c:
        fig_dist = PlotlyRenderer.distance_histogram(
            analysis,
            title=f"Edge Distances — {params['protein']}",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_d:
        fig_deg = PlotlyRenderer.degree_histogram(
            analysis,
            title=f"Degree Distribution — {params['protein']}",
        )
        st.plotly_chart(fig_deg, use_container_width=True)

    with st.expander("ℹ️ Edge Distance & Degree Distributions"):
        st.markdown("""
- **Edge Distance Distribution:** Shows the Euclidean distances of all
  edges. Most edges should be within the distance cutoff. The distribution
  peaks where residues are most commonly found — typically 3.8Å (Cα–Cα
  backbone distance) and 5–8Å (helical/sheet contacts).

- **Degree Distribution:** Shows how many neighbors each node has. A
  narrow distribution means all nodes have similar connectivity. A wide
  distribution means some residues (e.g., in a protein core) are much
  more connected than others (e.g., on the surface).
        """)

    st.markdown("---")

    # ── Feature matrix details ──
    st.markdown("### Node Feature Dimensions")
    st.markdown(f"""
| Feature | Dimensions | Description |
|---------|-----------|-------------|
| One-hot encoding | 20 | Amino acid identity (A, C, D, ..., Y) |
| Hydrophobicity | 1 | Kyte-Doolittle scale, normalized |
| Charge | 1 | Net charge at pH 7, normalized |
| Molecular weight | 1 | Da, normalized |
| Helix propensity | 1 | Chou-Fasman scale, normalized |
| **Total** | **{stats.feature_dim}** | |
    """)

    with st.expander("ℹ️ About Node Features"):
        st.markdown("""
Each node (residue) carries a **feature vector** of dimension 24:

1. **One-hot encoding (20D):** Which of the 20 standard amino acids is
   this residue? This is the most basic identity feature.
2. **Hydrophobicity (1D):** The Kyte-Doolittle scale measures how much
   each amino acid "likes" to be buried inside the protein (hydrophobic)
   vs. exposed to water (hydrophilic).
3. **Charge (1D):** Net charge at pH 7. Positively charged (K, R) and
   negatively charged (D, E) residues drive electrostatic interactions.
4. **Molecular weight (1D):** The physical size of the amino acid.
5. **Helix propensity (1D):** The Chou-Fasman scale measures how likely
   each amino acid is to form an α-helix.

These features are **normalized to [0, 1]** (except one-hot) so that
the GNN can learn from them effectively.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 3 — The k Slider
# ═══════════════════════════════════════════════════════════════════════


def page_k_slider() -> None:
    """Interactive k-sweep visualization."""
    st.title("🎚️ The k Slider")
    st.markdown(
        "Slide **k** and watch the protein graph evolve. At **k=2**, only backbone "
        "neighbors are connected. At **k=20**, the graph captures all the folding "
        "information — including long-range tertiary contacts."
    )
    st.markdown("---")

    _init_shared_state()

    # Protein selector in sidebar (shared keys so selection persists across pages)
    st.sidebar.markdown("### Graph Parameters")
    protein_name = st.sidebar.selectbox(
        "Protein Structure",
        _protein_options(),
        key="shared_protein",
        help="Choose a preset protein (or your uploaded PDB) for the k-sweep.",
    )

    cutoff = st.sidebar.slider(
        "Distance cutoff (Å)",
        5.0, 30.0, step=0.5,
        key="shared_cutoff",
        help="Maximum edge distance.",
    )

    # Main k slider
    k_val = st.slider(
        "k (nearest neighbors)",
        min_value=2, max_value=25, value=10, step=1,
        help="Slide to change the number of k-nearest neighbors. "
             "Watch how the graph topology changes!",
    )

    with st.spinner(f"Building graph at k={k_val}..."):
        analysis = _get_analysis(protein_name, k_val, cutoff)

    graph = analysis.graph
    stats = analysis.statistics

    # ── Metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("k", f"{k_val}")
    col2.metric("Edges", f"{stats.n_edges:,}")
    col3.metric("Density", f"{stats.density:.4f}")
    col4.metric("Mean Degree", f"{stats.mean_degree:.1f}")
    lr_frac = stats.n_long_range / max(stats.n_edges, 1) * 100
    col5.metric("Long-Range %", f"{lr_frac:.1f}%")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **k:** The number of nearest neighbors used for edge construction.
  Each residue is connected to its k closest Cα atoms in 3-D space.
- **Edges:** Total number of connections in the graph at the current k.
  More edges means a denser, more information-rich graph.
- **Density:** Fraction of all possible edges that are present.
  Density = E / (N × (N−1)). Even at high k, protein graphs remain sparse.
- **Mean Degree:** Average number of neighbors per node. Roughly
  equal to k, but can be lower if the distance cutoff prunes edges.
- **Long-Range %:** Percentage of edges connecting residues with
  sequence distance |i−j| > 12. These are tertiary contacts — the
  most informative edges for structure prediction and GNN learning.
        """)

    with st.expander("ℹ️ What changes as k increases?"):
        st.markdown("""
As you increase k:

1. **More edges** are added — the graph becomes denser
2. **Short-range contacts** saturate first (there are only ~4 backbone neighbors)
3. **Medium-range contacts** appear at k ≈ 6–8
4. **Long-range contacts** (tertiary) appear at k ≈ 10–15
5. **The graph captures more folding information** but becomes computationally heavier

**The sweet spot** for most GNN applications is k = 10–15: enough long-range
contacts to capture the fold, but sparse enough for efficient message passing.

**Try it:** Slide k from 2 to 20 and watch the "Long-Range %" metric change.
At k=4, it should be near 0%. At k=20, it captures the tertiary structure.
        """)

    st.markdown("---")

    # ── 3-D Graph at current k ──
    st.markdown(f"### Protein Graph at k = {k_val}")
    fig_3d = PlotlyRenderer.graph_3d(
        graph,
        color_by="contact_type",
        title=f"{protein_name} — k={k_val}",
        height=750,
        camera_distance=3.5,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    with st.expander("ℹ️ What to look for"):
        st.markdown("""
- At **k = 2–4:** Only grey (backbone) and blue (short-range) edges.
  The graph traces the peptide chain with minimal cross-links.
- At **k = 6–8:** Orange (medium-range) edges appear, connecting
  residues across turns and small loops.
- At **k = 10–15:** Red (long-range) edges emerge, connecting residues
  that are far apart in sequence but close in 3-D space.
- At **k = 20+:** The graph is dense — almost every nearby residue is
  connected. This captures maximum structural information but increases
  computational cost.

For **α-helices**, you'll see the edges form a helical pattern.
For **β-barrels**, long-range edges connect opposite sides of the barrel.
For **two-domain proteins**, you'll see dense clusters within each domain
and sparse connections between them.
        """)

    st.markdown("---")

    # ── Full k-sweep curves ──
    st.markdown("### k-Sweep Analysis")

    with st.spinner("Running k-sweep..."):
        sweep = _get_k_sweep(protein_name, cutoff)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_edges = PlotlyRenderer.k_sweep_edges(
            sweep,
            title=f"Edges vs. k — {protein_name}",
        )
        st.plotly_chart(fig_edges, use_container_width=True)

    with col_b:
        fig_density = PlotlyRenderer.k_sweep_density(
            sweep,
            title=f"Density vs. k — {protein_name}",
        )
        st.plotly_chart(fig_density, use_container_width=True)

    fig_lr = PlotlyRenderer.k_sweep_long_range(
        sweep,
        title=f"Long-Range Contact Fraction vs. k — {protein_name}",
    )
    st.plotly_chart(fig_lr, use_container_width=True)

    with st.expander("ℹ️ Interpreting the k-Sweep Curves"):
        st.markdown("""
These curves show how graph properties change as k increases:

1. **Edges vs. k:** Approximately linear growth. Each residue gets
   k neighbors, so total edges ≈ N × k (minus duplicates and cutoff).

2. **Density vs. k:** Density = E / (N(N−1)). It grows but remains small
   for large proteins — protein graphs are naturally sparse.

3. **Long-Range Contacts vs. k:** This is the key curve. It shows what
   fraction of edges are tertiary contacts (|i−j| > 12). A higher
   percentage means the graph captures more folding information.

**Key insight:** The long-range fraction typically plateaus above k ≈ 15.
Beyond that, you're adding redundant edges without gaining structural
information. This is why k = 10–15 is the practical sweet spot.
        """)

    st.markdown("---")

    # ── k-sweep table ──
    st.markdown("### k-Sweep Summary Table")

    import pandas as pd
    sweep_data = []
    for i, k in enumerate(sweep.k_values):
        s = sweep.statistics[i]
        sweep_data.append({
            "k": k,
            "Edges": s.n_edges,
            "Density": f"{s.density:.4f}",
            "Mean Degree": f"{s.mean_degree:.1f}",
            "Short-Range": s.n_short_range,
            "Medium-Range": s.n_medium_range,
            "Long-Range": s.n_long_range,
            "LR %": f"{sweep.long_range_fractions[i]*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(sweep_data), use_container_width=True)

    with st.expander("ℹ️ About this table"):
        st.markdown("""
This table shows exact values for each k in the sweep:

- **Short-Range:** Edges with |i−j| ≤ 4 (backbone + α-helix contacts)
- **Medium-Range:** Edges with 5 ≤ |i−j| ≤ 12 (turns, loops)
- **Long-Range:** Edges with |i−j| > 12 (tertiary contacts)
- **LR %:** Long-range fraction of total edges

Notice how short-range contacts saturate quickly (there are only ~4
backbone neighbors per residue), while long-range contacts keep growing.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 4 — The Contact Map
# ═══════════════════════════════════════════════════════════════════════


def page_contact_map() -> None:
    """Contact map and feature analysis."""
    st.title("📋 The Contact Map")
    st.markdown(
        "Visualize the **adjacency matrix**, **contact map** (colored by sequence "
        "distance), and the **node feature matrix** that would be fed to a GNN."
    )
    st.markdown("---")

    params = _graph_sidebar()

    with st.spinner("Building graph..."):
        analysis = _get_analysis(params["protein"], params["k"], params["cutoff"])

    graph = analysis.graph

    # ── Summary ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", f"{analysis.statistics.n_nodes}")
    col2.metric("Edges", f"{analysis.statistics.n_edges:,}")
    col3.metric("Feature Dim", f"{analysis.statistics.feature_dim}")
    col4.metric("Density", f"{analysis.statistics.density:.4f}")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
- **Nodes:** Number of amino acid residues (Cα atoms) in the protein.
- **Edges:** Number of connections in the graph. Each edge represents
  a spatial proximity relationship between two residues.
- **Feature Dim:** Dimensionality of each node's feature vector (24D:
  20 one-hot + hydrophobicity + charge + weight + helix propensity).
- **Density:** Fraction of all possible edges that are present.
  Density = E / (N × (N−1)). Protein graphs are typically sparse.
        """)

    # Maximum residues for N×N heatmaps before hitting Streamlit's
    # message-size limit (~200 MB).  A 500×500 heatmap is safe;
    # larger proteins would produce multi-hundred-MB payloads.
    _MAX_HEATMAP_N = 500

    st.markdown("---")

    # ── Adjacency Matrix ──
    st.markdown("### Adjacency Matrix")
    if analysis.statistics.n_nodes <= _MAX_HEATMAP_N:
        fig_adj = PlotlyRenderer.adjacency_heatmap(
            graph,
            title=f"Adjacency Matrix — {params['protein']} (k={params['k']})",
        )
        st.plotly_chart(fig_adj, use_container_width=True)
    else:
        st.warning(
            f"⚠️ The adjacency matrix is too large to display for this "
            f"protein ({analysis.statistics.n_nodes} residues). "
            f"The N×N heatmap ({analysis.statistics.n_nodes}×"
            f"{analysis.statistics.n_nodes} = "
            f"{analysis.statistics.n_nodes**2:,} cells) would exceed "
            f"Streamlit's message-size limit. Try a smaller protein "
            f"(≤{_MAX_HEATMAP_N} residues) or use one of the preset structures."
        )

    with st.expander("ℹ️ Reading the Adjacency Matrix"):
        st.markdown("""
The **adjacency matrix** A is an N×N binary matrix where:
- A[i,j] = 1 if there is an edge from residue i to residue j
- A[i,j] = 0 otherwise

**What to look for:**
- **Diagonal band:** Backbone and short-range contacts form a band
  near the diagonal (residues close in sequence are close in space).
- **Off-diagonal dots:** These are **long-range contacts** — residues
  far apart in sequence that are close in 3-D space. These are the
  structural "surprises" that define the protein fold.
- **Block structure:** In multi-domain proteins, you'll see dense blocks
  (one per domain) with sparse connections between them.
        """)

    st.markdown("---")

    # ── Contact Map ──
    st.markdown("### Contact Map (sequence distance)")
    if analysis.statistics.n_nodes <= _MAX_HEATMAP_N:
        fig_cmap = PlotlyRenderer.contact_map(
            graph,
            title=f"Contact Map — {params['protein']}",
        )
        st.plotly_chart(fig_cmap, use_container_width=True)
    else:
        st.warning(
            f"⚠️ The contact map is too large to display for this "
            f"protein ({analysis.statistics.n_nodes} residues). "
            f"The N×N heatmap ({analysis.statistics.n_nodes}×"
            f"{analysis.statistics.n_nodes} = "
            f"{analysis.statistics.n_nodes**2:,} cells) would exceed "
            f"Streamlit's message-size limit. Try a smaller protein "
            f"(≤{_MAX_HEATMAP_N} residues) or use one of the preset structures."
        )

    with st.expander("ℹ️ Reading the Contact Map"):
        st.markdown("""
The **contact map** is like the adjacency matrix, but colored by
**sequence distance** |i − j|:

- **Dark (low sequence distance):** Backbone and short-range contacts
  near the diagonal.
- **Bright (high sequence distance):** Long-range tertiary contacts
  far from the diagonal.

Contact maps are a standard tool in structural biology:
- **α-helices** appear as bands parallel to the diagonal (at offset ~4)
- **β-sheets** appear as anti-diagonal streaks (anti-parallel strands)
- **Domain boundaries** appear as gaps between dense blocks
        """)

    st.markdown("---")

    # ── Feature heatmap ──
    st.markdown("### Node Feature Matrix")

    feature_analysis = analyze_features(graph)
    fig_feat = PlotlyRenderer.feature_heatmap(
        feature_analysis,
        title=f"Node Feature Matrix — {params['protein']}",
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    with st.expander("ℹ️ Reading the Feature Matrix"):
        st.markdown("""
The **node feature matrix** X is an N×D matrix where:
- Rows are residues (N residues)
- Columns are features (D features)

**Features (left to right):**
1. **AA_A through AA_Y (20 columns):** One-hot encoding. Each residue
   has exactly one "1" in its amino acid column.
2. **Hydrophobicity (1 column):** Normalized Kyte-Doolittle scale.
   Bright = hydrophobic (interior), dark = hydrophilic (surface).
3. **Charge (1 column):** Normalized charge at pH 7.
4. **Weight (1 column):** Normalized molecular weight.
5. **Helix propensity (1 column):** How likely this amino acid is to
   form an α-helix.

This is the input tensor X that gets fed into the GNN's first layer.
        """)

    st.markdown("---")

    # ── Hydrophobicity profile ──
    st.markdown("### Hydrophobicity Profile")
    fig_hydro = PlotlyRenderer.hydrophobicity_profile(
        feature_analysis,
        title=f"Hydrophobicity — {params['protein']}",
    )
    st.plotly_chart(fig_hydro, use_container_width=True)

    with st.expander("ℹ️ About the Hydrophobicity Profile"):
        st.markdown("""
The **Kyte-Doolittle hydrophobicity scale** assigns each amino acid
a value:
- **Positive (red bars):** Hydrophobic — prefers the protein interior.
  Examples: Ile (4.5), Val (4.2), Leu (3.8)
- **Negative (blue bars):** Hydrophilic — prefers the protein surface.
  Examples: Arg (−4.5), Lys (−3.9), Asp (−3.5)

In globular proteins, hydrophobic residues tend to cluster in the
core, while hydrophilic residues face the aqueous environment.
This is the **hydrophobic effect** — the primary driving force of
protein folding.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 5 — Protein Comparison
# ═══════════════════════════════════════════════════════════════════════


def page_protein_comparison() -> None:
    """Compare all preset proteins (and the uploaded PDB, if any)."""
    st.title("📊 Protein Comparison")
    st.markdown(
        "Compare all six preset protein structures side by side. "
        "Different structural motifs produce fundamentally different "
        "graph topologies."
    )

    uploaded = _get_uploaded_protein()
    if uploaded is not None:
        st.info(
            f"Your uploaded protein **{uploaded.name}** "
            f"({uploaded.n_residues} residues) is included in the comparison below."
        )

    st.markdown("---")

    _init_shared_state()

    st.sidebar.markdown("### Graph Parameters")
    k = st.sidebar.slider(
        "k (neighbors)",
        2, 30, step=1,
        key="shared_k",
        help="k-NN parameter for all proteins.",
    )
    cutoff = st.sidebar.slider(
        "Distance cutoff (Å)",
        5.0, 30.0, step=0.5,
        key="shared_cutoff",
        help="Edge distance cutoff.",
    )

    with st.spinner("Analysing all presets..."):
        comparison = _cached_preset_comparison(k, cutoff)

    # If the user uploaded a protein, add it to the comparison
    if uploaded is not None:
        uploaded_analysis = _get_analysis(_UPLOADED_PROTEIN_LABEL, k, cutoff)
        comparison.analyses[uploaded.name] = uploaded_analysis
        # Append an entry matching the format used by compare_preset_proteins
        s = uploaded_analysis.statistics
        total = max(s.n_edges, 1)
        comparison.summary_table.append({
            "Protein": uploaded.name,
            "Residues": s.n_nodes,
            "Edges": s.n_edges,
            "Density": f"{s.density:.4f}",
            "Mean Degree": f"{s.mean_degree:.1f}",
            "Short-Range %": f"{s.n_short_range / total * 100:.1f}",
            "Medium-Range %": f"{s.n_medium_range / total * 100:.1f}",
            "Long-Range %": f"{s.n_long_range / total * 100:.1f}",
            "Feature Dim": s.feature_dim,
        })

    # ── Summary table ──
    st.markdown("### Summary Table")
    import pandas as pd
    df = pd.DataFrame(comparison.summary_table)
    st.dataframe(df, use_container_width=True)

    with st.expander("ℹ️ About the Summary Table"):
        st.markdown("""
This table compares all six preset protein structures at the same
k and distance cutoff:

- **Residues:** Total amino acids in the structure.
- **Edges:** Total edges in the k-NN graph.
- **Density:** Fraction of possible edges present.
- **Mean Degree:** Average neighbors per residue.
- **Short/Medium/Long-Range %:** Contact type breakdown.

**Key observations:**
- **α-Helix:** Dominated by short-range contacts (helical periodicity)
- **β-Sheet:** More medium-range contacts (strand-strand hydrogen bonds)
- **β-Barrel:** Highest long-range % (contacts across the barrel)
- **Random Coil:** Unpredictable contact pattern
- **Two-Domain:** Block structure with inter-domain contacts
        """)

    st.markdown("---")

    # ── Comparison bar charts ──
    st.markdown("### Comparison Charts")
    fig_bars = PlotlyRenderer.preset_comparison_bars(
        comparison,
        title=f"Preset Comparison (k={k})",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    with st.expander("ℹ️ Interpreting the Charts"):
        st.markdown("""
Three comparison metrics:

1. **Edge Count:** More residues → more edges, but the relationship
   depends on protein compactness. Globular proteins have more edges
   per residue than extended structures.

2. **Density (×10³):** Normalized for protein size. High density
   indicates a compact structure where many residues are within the
   distance cutoff.

3. **Long-Range %:** The most informative metric. Structures with
   complex 3-D folds (barrels, domains) have more long-range contacts
   than simple secondary structures (helices, sheets).
        """)

    st.markdown("---")

    # ── Individual graphs ──
    st.markdown("### Individual Protein Graphs")

    preset_names = list(comparison.analyses.keys())
    cols = st.columns(2)

    for idx, name in enumerate(preset_names):
        a = comparison.analyses[name]
        with cols[idx % 2]:
            fig = PlotlyRenderer.graph_3d(
                a.graph,
                color_by="contact_type",
                title=f"{name} (k={k})",
                node_size=4.0,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"{a.statistics.n_nodes} residues, "
                f"{a.statistics.n_edges} edges, "
                f"density={a.statistics.density:.4f}"
            )

    with st.expander("ℹ️ Comparing the Graphs"):
        st.markdown("""
Each graph shows a different structural motif:

- **α-Helix:** A spiral with regular short-range contacts (blue edges at i+3, i+4).
- **β-Sheet:** Flat strands with medium-range contacts between adjacent strands.
- **Helix-Turn-Helix:** Two helical regions connected by a short turn.
- **β-Barrel:** A cylinder of β-strands — rich in long-range contacts.
- **Random Coil:** An unstructured chain — contact pattern is stochastic.
- **Two-Domain:** Two compact clusters connected by a flexible linker.

Notice how the **edge coloring** reveals the structural hierarchy:
grey (backbone) → blue (helix) → orange (turns) → red (tertiary).
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Page 6 — Theory & Mathematics
# ═══════════════════════════════════════════════════════════════════════


def page_theory() -> None:
    """Theory and mathematics page."""
    st.title("📚 Theory & Mathematics")
    st.markdown(
        "The mathematical framework behind protein graph construction "
        "and Graph Neural Network featurization."
    )
    st.markdown("---")

    # ── 1. Graph Representation ──
    with st.expander("1. Graph Representation of Proteins"):
        st.markdown(r"""
### Graph Representation

A protein structure is represented as a graph $G = (V, E, X, F)$:

- **$V$:** Set of $N$ nodes (amino acid residues at Cα positions)
- **$E$:** Set of edges connecting spatially proximal residues
- **$X \in \mathbb{R}^{N \times D}$:** Node feature matrix
- **$F \in \mathbb{R}^{|E| \times D_e}$:** Edge feature matrix

The graph is constructed from 3-D coordinates via the **k-Nearest
Neighbors (k-NN)** algorithm: for each residue $i$, we connect it
to its $k$ nearest Cα neighbors in Euclidean space, subject to a
distance cutoff $r_{\max}$.

$$E = \{(i, j) : j \in \text{kNN}(i) \text{ and } d_{ij} \leq r_{\max}\}$$
        """)

    # ── 2. k-NN with KD-Tree ──
    with st.expander("2. k-Nearest Neighbors and KD-Tree"):
        st.markdown(r"""
### k-NN Edge Construction

Given $N$ residues with Cα coordinates $\{\mathbf{r}_i\}_{i=1}^N$:

**1. Build a KD-Tree** for $O(N \log N)$ spatial indexing.

**2. For each node $i$, query the $k$ nearest neighbors:**

$$d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2$$

**3. Apply a distance cutoff** to remove spurious long-distance edges:
        """)
        st.latex(r"A_{ij} = \begin{cases} 1 & \text{if } j \in \text{kNN}(i) \text{ and } d_{ij} \leq r_{\max} \\ 0 & \text{otherwise} \end{cases}")
        st.markdown(r"""
**Complexity:** $O(N \log N)$ for KD-Tree construction +
$O(N \cdot k \cdot \log N)$ for queries.

**Note:** The graph is generally **directed** (j ∈ kNN(i) does not
imply i ∈ kNN(j)), though it can be symmetrized.
        """)

    # ── 3. Node Features ──
    with st.expander("3. Node Featurization"):
        st.markdown(r"""
### Node Feature Construction

Each node $i$ carries a feature vector $\mathbf{x}_i \in \mathbb{R}^D$:
        """)
        st.latex(r"\mathbf{x}_i = [\mathbf{e}_{aa(i)},\; h_i^{\text{norm}},\; q_i^{\text{norm}},\; w_i^{\text{norm}},\; p_i^{\text{norm}}]")
        st.markdown(r"""
where:

| Feature | Symbol | Dimensions | Description |
|---------|--------|-----------|-------------|
| One-hot encoding | eₐₐ | 20 | Indicator vector for amino acid identity |
| Hydrophobicity | hᵢ | 1 | Kyte-Doolittle scale, min-max normalized |
| Charge | qᵢ | 1 | Net charge at pH 7, normalized |
| Molecular weight | wᵢ | 1 | Daltons, normalized |
| Helix propensity | pᵢ | 1 | Chou-Fasman scale, normalized |

**Normalization:**
        """)
        st.latex(r"h_i^{\text{norm}} = \frac{h_i - h_{\min}}{h_{\max} - h_{\min}}")
        st.markdown(r"""
Total feature dimension: $D = 24$.
        """)

    # ── 4. Edge Features ──
    with st.expander("4. Edge Featurization"):
        st.markdown(r"""
### Edge Feature Construction

Each edge $(i, j) \in E$ carries a feature vector
$\mathbf{f}_{ij} \in \mathbb{R}^{D_e}$:

**1. Euclidean distance:**
$$d_{ij} = \|\mathbf{r}_j - \mathbf{r}_i\|_2$$

**2. Direction vector (unit):**
$$\hat{\mathbf{r}}_{ij} = \frac{\mathbf{r}_j - \mathbf{r}_i}{d_{ij}} \in \mathbb{R}^3$$

**3. Sequence distance:**
$$s_{ij} = |i - j|$$

**4. Orientation quaternion:**
The quaternion $\mathbf{q}_{ij} = (w, x, y, z) \in \mathbb{R}^4$
represents the rotation from the z-axis $\hat{\mathbf{z}}$ to the
direction $\hat{\mathbf{r}}_{ij}$:
        """)
        st.latex(r"\mathbf{q} = \frac{1}{\|\mathbf{q}'\|} \begin{pmatrix} 1 + \hat{\mathbf{z}} \cdot \hat{\mathbf{r}}_{ij} \\ \hat{\mathbf{z}} \times \hat{\mathbf{r}}_{ij} \end{pmatrix}")
        st.markdown(r"""
Quaternions provide a **singularity-free** 4-D representation of
orientation, avoiding the gimbal lock problem of Euler angles.
        """)

    # ── 5. Adjacency Matrix ──
    with st.expander("5. Sparse Adjacency Matrix"):
        st.markdown(r"""
### Adjacency Matrix

The adjacency matrix $A \in \{0, 1\}^{N \times N}$ encodes the
graph topology:
        """)
        st.latex(r"A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}")
        st.markdown(r"""
**Properties:**
- **Sparsity:** Graph density $\rho = |E| \;/\; N(N-1) \ll 1$
  for protein graphs (typically $\rho < 0.05$).
- **Degree:** deg$(i) = \sum_j A_{ij}$
- **Laplacian:** $L = D - A$ where $D$ = diag(deg)

The adjacency matrix is stored as an **edge index** (COO format)
$(2, |E|)$ for efficient sparse operations:
        """)
        st.latex(r"\text{edge\_index} = \begin{pmatrix} i_1 & i_2 & \cdots & i_{|E|} \\ j_1 & j_2 & \cdots & j_{|E|} \end{pmatrix}")

    # ── 6. Contact Classification ──
    with st.expander("6. Contact Classification by Sequence Distance"):
        st.markdown(r"""
### Contact Classification

Edges are classified by **sequence distance** $s_{ij} = |i - j|$:

| Category | Range | Structural Meaning |
|----------|-------|-------------------|
| Backbone | $s = 1$ | Peptide bond |
| Short-range | $2 \leq s \leq 4$ | α-helix H-bonds ($i \to i+4$) |
| Medium-range | $5 \leq s \leq 12$ | Turns, loops, β-hairpins |
| Long-range | $s > 12$ | Tertiary contacts |

**Why this matters for GNNs:**

Short-range contacts encode **secondary structure** (helix, sheet).
Long-range contacts encode **tertiary structure** (the 3-D fold).
A GNN must learn to propagate information along **both** types of edges.

The fraction of long-range contacts is a key metric:
$$f_{\text{LR}} = \frac{|\{(i,j) \in E : |i-j| > 12\}|}{|E|}$$
        """)

    # ── 7. GNN Message Passing ──
    with st.expander("7. GNN Message Passing Framework"):
        st.markdown(r"""
### Message Passing Neural Networks

A Graph Neural Network processes the protein graph through
**message passing** layers:

**Message computation:**
        """)
        st.latex(r"\mathbf{m}_{ij}^{(l)} = \phi^{(l)}\!\left( \mathbf{h}_i^{(l)},\; \mathbf{h}_j^{(l)},\; \mathbf{f}_{ij} \right)")
        st.markdown("**Aggregation:**")
        st.latex(r"\mathbf{M}_i^{(l)} = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)}")
        st.markdown(r"""
where $\bigoplus$ is a permutation-invariant function (sum, mean, max).

**Node update:**
        """)
        st.latex(r"\mathbf{h}_i^{(l+1)} = \psi^{(l)}\!\left( \mathbf{h}_i^{(l)},\; \mathbf{M}_i^{(l)} \right)")
        st.markdown(r"""
**Initial features:** $\mathbf{h}_i^{(0)} = \mathbf{x}_i$ (our
node feature vector from step 3).

After $L$ layers, each node's embedding $\mathbf{h}_i^{(L)}$
incorporates information from its $L$-hop neighborhood — this is
how the GNN "sees" the protein structure.
        """)

    # ── 8. PyTorch Geometric Data Object ──
    with st.expander("8. PyTorch Geometric Data Object"):
        st.markdown(r"""
### The `Data` Object

The final output of the pipeline is a PyTorch Geometric `Data`
object containing:

```python
Data(
    x       = [N, 24],       # Node features
    edge_index = [2, E],     # COO edge indices
    edge_attr  = [E, 9],     # Edge features (d, r̂, s, q)
    pos     = [N, 3],        # Cα coordinates
    y       = ...,           # Labels (task-dependent)
)
```

**Edge features breakdown (9 dims):**
| Dims | Feature |
|------|---------|
| 1 | Euclidean distance $d_{ij}$ |
| 3 | Direction vector $\hat{\mathbf{r}}_{ij}$ |
| 1 | Sequence distance $s_{ij}$ |
| 4 | Orientation quaternion $\mathbf{q}_{ij}$ |

This representation is directly consumable by GNN architectures
such as SchNet, DimeNet, GVP-GNN, and EquiFormer.
        """)

    # ── 9. KD-Tree Algorithm ──
    with st.expander("9. KD-Tree Algorithm"):
        st.markdown(r"""
### KD-Tree for Spatial Queries

A **KD-Tree** (k-dimensional tree) is a space-partitioning data
structure for efficient nearest-neighbor queries in $\mathbb{R}^d$.

**Construction ($O(N \log N)$):**
1. Choose the axis with greatest spread
2. Find the median along that axis
3. Split points at the median
4. Recursively build left and right subtrees

**Query ($O(\log N)$ average):**
1. Traverse the tree to the leaf containing the query point
2. Backtrack, pruning branches that cannot contain closer points
3. Return the $k$ nearest neighbors

**For proteins:** We build a 3-D KD-Tree over the Cα coordinates,
then query each residue for its $k$ nearest neighbors. This reduces
the naive $O(N^2)$ all-pairs distance computation to $O(N \log N)$.
        """)

    # ── 10. Quaternion Representation ──
    with st.expander("10. Quaternion Orientation Encoding"):
        st.markdown(r"""
### Quaternions for Edge Orientation

A **quaternion** $\mathbf{q} = w + xi + yj + zk$ is a 4-D
hypercomplex number used to represent 3-D rotations:
        """)
        st.latex(r"\mathbf{q} = \left(\cos\frac{\theta}{2},\; \hat{\mathbf{u}} \sin\frac{\theta}{2}\right)")
        st.markdown(r"""
where $\theta$ is the rotation angle and $\hat{\mathbf{u}}$ is the
rotation axis.

**Advantages over Euler angles:**
- No gimbal lock (singularity-free)
- Smooth interpolation (SLERP)
- Compact 4-D representation
- Easy composition: $\mathbf{q}_{AC} = \mathbf{q}_{AB} \otimes \mathbf{q}_{BC}$

**For protein edges:** The quaternion $\mathbf{q}_{ij}$ encodes the
rotation from the z-axis to the edge direction $\hat{\mathbf{r}}_{ij}$.
This captures the spatial **orientation** of each edge, not just its
distance — critical for equivariant GNNs.
        """)

    # ── 11. Applications ──
    with st.expander("11. Applications in Geometric Deep Learning"):
        st.markdown(r"""
### Applications

Protein graphs are used in cutting-edge AI models:

| Model | Task | Graph Features Used |
|-------|------|-------------------|
| AlphaFold2 | Structure prediction | Pair representations, spatial features |
| ESMFold | Single-sequence folding | Residue graphs, attention |
| RoseTTAFold | Co-evolution + structure | 3-track with graph transformer |
| GVP-GNN | Function prediction | Scalar + vector features on k-NN graphs |
| ProteinMPNN | Inverse folding (design) | Message passing on backbone graph |
| DiffDock | Molecular docking | Protein-ligand interaction graph |

**This project** builds the foundational pipeline that all these models
use: converting 3-D coordinates into graph-structured data with
learnable node and edge features.

The graph representation is the **lingua franca** of geometric deep
learning for proteins.
        """)

    # ── 12. References ──
    with st.expander("12. References"):
        st.markdown(r"""
### References

1. **Bronstein, M. M., Bruna, J., Cohen, T. & Veličković, P.** (2021).
   Geometric deep learning: Grids, groups, graphs, geodesics, and gauges.
   *arXiv preprint arXiv:2104.13478*.

2. **Jumper, J. et al.** (2021). Highly accurate protein structure
   prediction with AlphaFold. *Nature*, 596, 583–589.

3. **Jing, B., Eismann, S., Suriana, P., Townshend, R. J. L. & Dror, R.**
   (2021). Learning from protein structure with geometric vector
   perceptrons. *ICLR*.

4. **Satorras, V. G., Hoogeboom, E. & Welling, M.** (2021). E(n)
   equivariant graph neural networks. *ICML*.

5. **Friedman, J. H., Bentley, J. L. & Finkel, R. A.** (1977). An
   algorithm for finding best matches in logarithmic expected time.
   *ACM Transactions on Mathematical Software*, 3(3), 209–226.

6. **Kyte, J. & Doolittle, R. F.** (1982). A simple method for
   displaying the hydropathic character of a protein. *Journal of
   Molecular Biology*, 157(1), 105–132.

7. **Kuipers, J. B.** (1999). *Quaternions and Rotation Sequences*.
   Princeton University Press.

8. **Chou, P. Y. & Fasman, G. D.** (1978). Empirical predictions of
   protein conformation. *Annual Review of Biochemistry*, 47, 251–276.

9. **Ingraham, J., Garg, V. K., Barzilay, R. & Jaakkola, T.** (2019).
   Generative models for graph-based protein design. *NeurIPS*.

10. **Gainza, P. et al.** (2020). Deciphering interaction fingerprints
    from protein molecular surfaces using geometric deep learning.
    *Nature Methods*, 17, 184–192.

11. **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. &
    Dahl, G. E.** (2017). Neural message passing for quantum chemistry.
    *ICML*.

12. **Dauparas, J. et al.** (2022). Robust deep learning–based protein
    sequence design using ProteinMPNN. *Science*, 378(6615), 49–56.

13. **Corso, G., Stärk, H., Jing, B., Barzilay, R. & Jaakkola, T.**
    (2023). DiffDock: Diffusion steps, twists, and turns for molecular
    docking. *ICLR*.

14. **Schütt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A.
    & Müller, K.-R.** (2018). SchNet — A deep learning architecture for
    molecules and materials. *Journal of Chemical Physics*, 148, 241722.

15. **Klicpera, J., Groß, J. & Günnemann, S.** (2020). Directional
    message passing for molecular graphs. *ICLR*.
        """)

    st.markdown(FOOTER, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point."""
    page = render_sidebar()

    dispatch = {
        "🏠 Home": page_home,
        "🧠 The Neural View": page_neural_view,
        "🎚️ The k Slider": page_k_slider,
        "📋 The Contact Map": page_contact_map,
        "📊 Protein Comparison": page_protein_comparison,
        "📚 Theory & Mathematics": page_theory,
    }

    handler = dispatch.get(page, page_home)
    handler()


if __name__ == "__main__":
    main()
