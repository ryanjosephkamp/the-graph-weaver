# 🕸️ The Graph Weaver — Protein-to-Graph Conversion

**Week 14, Project 3** | Biophysics Portfolio  
**Ryan Kamp** | University of Cincinnati, Department of Computer Science  
kamprj@mail.uc.edu | [GitHub](https://github.com/ryanjosephkamp/the-graph-weaver)

---

## Overview

Proteins aren't images — they're irregular 3-D structures.
You can't feed them into a CNN. You feed them into a
**Graph Neural Network**. This project builds the pipeline
that converts atomic coordinates into a learnable graph.

This project implements protein-to-graph conversion for GNN
featurization, covering:

- **k-NN Graph Construction** — KD-Tree-based O(N log N) edge building
- **24-D Node Features** — one-hot residue type + hydrophobicity + charge + weight + helix propensity
- **9-D Edge Features** — distance + direction vector + sequence separation + orientation quaternion
- **Sparse Adjacency Matrix** — binary N×N connectivity encoding
- **Contact Classification** — backbone, short-range (α-helix), medium-range, long-range (tertiary)
- **k-Sweep Analysis** — graph topology as a function of neighborhood size
- **Six preset proteins** — α-helix, β-sheet, helix-turn-helix, β-barrel, random coil, two-domain

---

## Quick Start

```bash
# Navigate to the project directory
cd week_14_projects/week_14_project_3

# Activate the virtual environment
source ../../.venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Run the default analysis
python main.py

# Run the Streamlit dashboard
streamlit run app.py

# Run the test suite
pytest tests/ -v
```

---

## Project Structure

```
week_14_project_3/
├── src/
│   ├── __init__.py             # Package facade (re-exports all symbols)
│   ├── graph_engine.py         # Core graph construction engine (~1,200 lines)
│   ├── analysis.py             # Analysis pipelines (~550 lines)
│   └── visualization.py        # Plotly + Matplotlib rendering (~1,020 lines)
├── tests/
│   └── test_graph_weaver.py    # 20 classes, 122 tests (~960 lines)
├── docs/
│   ├── scientific_report.md    # Full scientific report
│   └── w14p3_graph_weaver_ieee.tex  # IEEE conference paper
├── figures/                    # Generated figures (auto-created)
├── main.py                     # CLI entry point (4 modes)
├── app.py                      # Streamlit dashboard (6 pages, ~1,580 lines)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## The Science

### Why Graphs?

Proteins are not grids. A CNN sees square pixels. A GNN sees
**nodes** (amino acids) and **edges** (spatial contacts). This
graph captures the 3-D topology that determines protein function.

### Key Equations

| Quantity | Equation |
|----------|----------|
| k-NN edge | (i, j) ∈ E if j ∈ k-closest(i) and d_ij < d_max |
| Euclidean distance | d_ij = ‖r_j − r_i‖ |
| Direction vector | r̂_ij = (r_j − r_i) / d_ij |
| Quaternion | q = (cos(θ/2), â·sin(θ/2)) |
| Graph density | ρ = \|E\| / (N(N−1)) |
| Mean degree | d̄ = 2\|E\| / N |

### Contact Classification

| Δ = \|i − j\| | Category | Structural role |
|----------------|----------|-----------------|
| ≤ 1 | Backbone | Peptide bond neighbors |
| 2–4 | Short-range | α-helix contacts (i→i+4) |
| 5–12 | Medium-range | Loops, turns, β-hairpins |
| > 12 | Long-range | Tertiary contacts (the hard part) |

### Six Preset Proteins

| Protein | N | Description |
|---------|---|-------------|
| α-Helix | 30 | 3.6 res/turn, rise 1.5 Å/res |
| β-Sheet | 32 | 4 strands × 8 residues |
| Helix-Turn-Helix | 34 | Two helices + 4-residue turn |
| β-Barrel | 48 | 8 strands × 6 residues, circular |
| Random Coil | 40 | Gaussian random walk |
| Two-Domain | 45 | Two domains + 5-residue linker |

---

## CLI Usage

```bash
# Default: analyze α-helix
python main.py

# Analyze a specific protein
python main.py --analyze --protein barrel --save --verbose

# Compare all six preset proteins
python main.py --compare --save

# k-sweep analysis
python main.py --sweep --protein helix --save

# Contact type analysis
python main.py --contacts --protein sheet --save

# Custom k and cutoff
python main.py --analyze -k 15 --cutoff 8.0 --verbose
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--analyze` | Standard graph analysis | ✓ |
| `--compare` | Compare all 6 presets | |
| `--sweep` | k-sweep analysis | |
| `--contacts` | Contact type analysis | |
| `--protein NAME` | Preset protein | helix |
| `-k` / `--k-neighbors` | Number of neighbors | 10 |
| `--cutoff` | Distance cutoff (Å) | 10.0 |
| `--save` | Save figures to `figures/` | |
| `--verbose` | Verbose output | |

---

## Streamlit Dashboard

```bash
streamlit run app.py
```

### Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview, key equations, graph preview, science dropdowns |
| 🧠 **Neural View** | Interactive 3-D graph with 4 edge-coloring modes (contact type, hydrophobicity, charge, residue index), contact-type breakdown (pie + histogram), edge distance & degree distributions, node feature table |
| 🎚️ **k Slider** | Interactive k-sweep with real-time graph reconstruction, edge/density/long-range plots, k-sweep summary table |
| 📋 **Contact Map** | Adjacency heatmap, contact map colored by sequence distance, node feature matrix heatmap, hydrophobicity profile |
| 📊 **Protein Comparison** | All 6 presets (+ uploaded PDB) compared side by side, bar charts, summary tables, individual 3-D graphs |
| 📚 **Theory & Mathematics** | 12 expandable sections: graph representation, k-NN & KD-Trees, node featurization, edge featurization, adjacency matrix, contact classification, GNN message passing, PyTorch Geometric data object, KD-Tree algorithm, quaternion orientation, applications in geometric deep learning, references |

### PDB Upload

Upload your own `.pdb` file via the sidebar to analyse any real protein structure. The Cα atoms are extracted automatically and the uploaded protein integrates into every page — Neural View, k Slider, Contact Map, and Protein Comparison.

Every visualization and metric panel includes an **ℹ️ informational dropdown** explaining what you're seeing, how to interpret the data, and why it matters — 35 informational expanders across the 6 pages.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run specific test class
pytest tests/test_graph_weaver.py::TestBuilders -v
```

### Test Coverage

- **20 test classes**, **122 test methods** covering all modules (~960 lines)
- Engine tests: Constants, Residue, ProteinStructure, Builders,
  PDBParsing, NodeFeatures, EdgeConstruction, Adjacency,
  GraphStatistics, ContactClassification, FullPipeline
- Analysis tests: AnalyzeGraph, KSweep, AnalyzeContacts, AnalyzeFeatures,
  PresetComparison, GraphSummary
- Visualization tests: PlotlyRenderer (13 methods), MatplotlibRenderer (6 methods)
- CLI tests: argument parsing for all modes and flags

---

## Dependencies

- **Python ≥ 3.10**
- **NumPy** — numerical computation
- **SciPy** — KD-Tree and spatial algorithms
- **Matplotlib** — static publication figures
- **Plotly** — interactive HTML visualization
- **Streamlit** — web dashboard
- **Pandas** — data tables
- **pytest** — testing framework

---

## References

1. Bronstein, M. M. et al. (2021). Geometric deep learning. *arXiv:2104.13478*.
2. Jumper, J. et al. (2021). *Nature*, 596, 583–589.
3. Jing, B. et al. (2021). Geometric vector perceptrons. *ICLR*.
4. Friedman, J. H. et al. (1977). *ACM Trans. Math. Softw.*, 3(3), 209–226.
5. Kyte, J. & Doolittle, R. F. (1982). *J. Mol. Biol.*, 157(1), 105–132.
6. Kuipers, J. B. (1999). *Quaternions and Rotation Sequences*. Princeton.

---

*Biophysics Portfolio — CS Research Self-Study — University of Cincinnati*
