# Week 14 - Project 3: "The Graph Weaver" – Protein-to-Graph Conversion

## Overview

**Week:** 14 (Apr 21 – Apr 28)  
**Theme:** Computational Geometry, Differential Geometry, and Graph Theory  
**Goal:** Implement the algorithms that turn "Atomic Coordinates" into "Learnable Features" (Curvature, SASA, Geodesics).

---

## Project Details

### The "Gap" It Fills
Mastery of **Graph Neural Networks (GNNs)** and **Tensor Featurization**.

This is the direct bridge to your AI work. How do you feed a protein into PyTorch? You don't use a CNN (proteins aren't square). You use a Graph.

You will build a pipeline that converts a `.pdb` file into a `geometric_data` object (Nodes, Edges, Attributes).

### The Concept
- **Nodes:** Each Amino Acid (Cα).
  - *Features:* One-hot encoding (A, C, D...), hydrophobicity, charge.
- **Edges:** Connect residues within 10Å (k-Nearest Neighbors).
  - *Features:* Distance (dij), Directional Vector (rij), Orientation Angles (quaternions).
- **Output:** A sparse Adjacency Matrix and Feature Matrix.

### Novelty/Creative Angle
**"The Neural View":**
- Visualize the graph *over* the protein.
- Draw the "Edges" as glowing lines connecting the residues.
- *The Twist:* Color the edges by "Sequence Distance" (how far apart they are in the string).
  - Short-range edges (i to i+4) = α-helices.
  - Long-range edges (i to i+50) = **Tertiary Contacts** (the hard part of folding).
- This visualizes exactly what a GNN "sees" when it predicts structure.

### Technical Implementation
- **Language:** Python (NetworkX, PyTorch Geometric).
- **Math:** k-NN algorithms (KD-Tree) for fast edge construction.

### The "Paper" & Interactive Element
- *Interactive:* "The k Slider." User changes k. At k=4, only local backbone links. At k=20, the graph becomes dense and captures all the folding information.
- *Paper Focus:* "Graph Neural Network Featurization of Protein Structures: Constructing Sparse Representations for Geometric Deep Learning."

---

## Progress Tracking

- [ ] Initial research and planning
- [ ] Core implementation
- [ ] Testing and validation
- [ ] Documentation and paper draft
- [ ] Interactive demo creation
