"""
The Graph Weaver — Core Graph Engine.

Converts protein structures (PDB files or synthetic coordinates)
into graph representations suitable for Graph Neural Networks.

Pipeline:
    PDB / coordinates
        → Residue extraction (Cα atoms)
        → Node feature construction (one-hot, hydrophobicity, charge)
        → k-NN edge construction (KD-Tree, distance cutoff)
        → Edge feature construction (distance, direction, quaternion)
        → Sparse adjacency matrix + feature matrices
        → PyTorch Geometric Data object
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_K: int = 10
"""Default number of nearest neighbors for k-NN graph."""

DEFAULT_DISTANCE_CUTOFF: float = 10.0
"""Default distance cutoff in Angstroms for edge construction."""

# Standard 20 amino acids (one-letter codes, alphabetical)
AMINO_ACIDS: List[str] = list("ACDEFGHIKLMNPQRSTVWY")

AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
"""Map one-letter amino acid code → index (0–19)."""

NUM_AMINO_ACIDS: int = len(AMINO_ACIDS)

# Three-letter to one-letter mapping
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

ONE_TO_THREE: Dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY: Dict[str, float] = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5, "F":  2.8,
    "G": -0.4, "H": -3.2, "I":  4.5, "K": -3.9, "L":  3.8,
    "M":  1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# Charge at pH 7.0
CHARGE: Dict[str, float] = {
    "A":  0.0, "C":  0.0, "D": -1.0, "E": -1.0, "F":  0.0,
    "G":  0.0, "H":  0.5, "I":  0.0, "K":  1.0, "L":  0.0,
    "M":  0.0, "N":  0.0, "P":  0.0, "Q":  0.0, "R":  1.0,
    "S":  0.0, "T":  0.0, "V":  0.0, "W":  0.0, "Y":  0.0,
}

# Molecular weight of amino acids (Da)
MOLECULAR_WEIGHT: Dict[str, float] = {
    "A":  89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G":  75.0, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.1, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

# Secondary structure propensities (Chou-Fasman scale, normalized)
HELIX_PROPENSITY: Dict[str, float] = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Residue:
    """A single amino acid residue."""
    index: int
    name: str              # One-letter code
    three_letter: str      # Three-letter code
    chain_id: str          # PDB chain identifier
    residue_number: int    # PDB residue sequence number
    ca_position: NDArray   # Cα (x, y, z) in Angstroms

    @property
    def position(self) -> NDArray:
        return self.ca_position

    @property
    def hydrophobicity(self) -> float:
        return HYDROPHOBICITY.get(self.name, 0.0)

    @property
    def charge(self) -> float:
        return CHARGE.get(self.name, 0.0)

    @property
    def weight(self) -> float:
        return MOLECULAR_WEIGHT.get(self.name, 0.0)

    @property
    def helix_propensity(self) -> float:
        return HELIX_PROPENSITY.get(self.name, 0.0)


@dataclass
class ProteinStructure:
    """A protein represented as a list of residues."""
    name: str
    residues: List[Residue]
    sequence: str = ""

    def __post_init__(self):
        if not self.sequence:
            self.sequence = "".join(r.name for r in self.residues)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def ca_coordinates(self) -> NDArray:
        """Return (N, 3) array of Cα positions."""
        return np.array([r.ca_position for r in self.residues])

    @property
    def chain_ids(self) -> List[str]:
        return list(set(r.chain_id for r in self.residues))


@dataclass
class NodeFeatures:
    """Node feature matrix for the protein graph.

    Attributes
    ----------
    one_hot : NDArray
        (N, 20) one-hot encoding of amino acid identity.
    hydrophobicity : NDArray
        (N,) Kyte-Doolittle hydrophobicity values.
    charge : NDArray
        (N,) Net charge at pH 7.
    weight : NDArray
        (N,) Molecular weight (Da).
    helix_propensity : NDArray
        (N,) Chou-Fasman helix propensity.
    feature_matrix : NDArray
        (N, D) concatenated feature matrix.
    feature_names : List[str]
        Names corresponding to each column of feature_matrix.
    """
    one_hot: NDArray
    hydrophobicity: NDArray
    charge: NDArray
    weight: NDArray
    helix_propensity: NDArray
    feature_matrix: NDArray
    feature_names: List[str]


@dataclass
class EdgeData:
    """Edge data for the protein graph.

    Attributes
    ----------
    edge_index : NDArray
        (2, E) array of [source, target] node indices.
    distances : NDArray
        (E,) Euclidean distance for each edge.
    direction_vectors : NDArray
        (E, 3) unit direction vectors r_ij / ||r_ij||.
    sequence_distances : NDArray
        (E,) |i - j| sequence separation for each edge.
    quaternions : NDArray
        (E, 4) orientation quaternions for each edge.
    n_edges : int
        Number of edges.
    """
    edge_index: NDArray
    distances: NDArray
    direction_vectors: NDArray
    sequence_distances: NDArray
    quaternions: NDArray
    n_edges: int


@dataclass
class AdjacencyData:
    """Sparse adjacency representation.

    Attributes
    ----------
    sparse_matrix : NDArray
        (N, N) adjacency matrix (0/1).
    degree : NDArray
        (N,) degree of each node.
    density : float
        Fraction of possible edges present.
    """
    sparse_matrix: NDArray
    degree: NDArray
    density: float


@dataclass
class ProteinGraph:
    """Complete protein graph representation.

    This is the primary output of the pipeline, containing
    all components needed for GNN input.
    """
    protein: ProteinStructure
    node_features: NodeFeatures
    edge_data: EdgeData
    adjacency: AdjacencyData
    k: int
    distance_cutoff: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.protein.name

    @property
    def n_nodes(self) -> int:
        return self.protein.n_residues

    @property
    def n_edges(self) -> int:
        return self.edge_data.n_edges

    @property
    def feature_dim(self) -> int:
        return self.node_features.feature_matrix.shape[1]


@dataclass
class GraphStatistics:
    """Summary statistics for a protein graph."""
    n_nodes: int
    n_edges: int
    density: float
    mean_degree: float
    std_degree: float
    min_degree: int
    max_degree: int
    mean_edge_distance: float
    std_edge_distance: float
    mean_sequence_distance: float
    std_sequence_distance: float
    n_short_range: int       # |i-j| <= 4
    n_medium_range: int      # 4 < |i-j| <= 12
    n_long_range: int        # |i-j| > 12
    feature_dim: int
    k: int
    distance_cutoff: float


# ═══════════════════════════════════════════════════════════════════════
# PDB parsing
# ═══════════════════════════════════════════════════════════════════════


def parse_pdb(pdb_text: str, name: str = "protein") -> ProteinStructure:
    """Parse PDB-format text and extract Cα atoms.

    Parameters
    ----------
    pdb_text : str
        Contents of a PDB file.
    name : str
        Name for the protein structure.

    Returns
    -------
    ProteinStructure
        Extracted residues with Cα coordinates.
    """
    residues: List[Residue] = []
    seen: set = set()

    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        res_name = line[17:20].strip()
        chain_id = line[21:22].strip() or "A"
        res_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        key = (chain_id, res_num)
        if key in seen:
            continue
        seen.add(key)

        one_letter = THREE_TO_ONE.get(res_name, "A")
        idx = len(residues)

        residues.append(Residue(
            index=idx,
            name=one_letter,
            three_letter=res_name,
            chain_id=chain_id,
            residue_number=res_num,
            ca_position=np.array([x, y, z], dtype=np.float64),
        ))

    return ProteinStructure(name=name, residues=residues)


def parse_pdb_file(filepath: str, name: str = "") -> ProteinStructure:
    """Parse a PDB file from disk.

    Parameters
    ----------
    filepath : str
        Path to a .pdb file.
    name : str
        Optional name override.

    Returns
    -------
    ProteinStructure
    """
    with open(filepath, "r") as f:
        text = f.read()
    if not name:
        import os
        name = os.path.splitext(os.path.basename(filepath))[0]
    return parse_pdb(text, name=name)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic protein builders
# ═══════════════════════════════════════════════════════════════════════


def build_alpha_helix(
    n_residues: int = 30,
    name: str = "α-Helix",
) -> ProteinStructure:
    """Build a synthetic α-helix backbone (3.6 residues/turn).

    Parameters
    ----------
    n_residues : int
        Number of residues.
    name : str
        Structure name.

    Returns
    -------
    ProteinStructure
    """
    # α-helix parameters: rise = 1.5Å, 3.6 residues/turn, radius ≈ 2.3Å
    rise_per_residue = 1.5
    residues_per_turn = 3.6
    radius = 2.3

    residues = []
    # Use a repeating sequence for variety
    helix_seq = "AELKAIAQELKAIAQA"

    for i in range(n_residues):
        angle = 2 * math.pi * i / residues_per_turn
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = rise_per_residue * i

        aa = helix_seq[i % len(helix_seq)]
        residues.append(Residue(
            index=i,
            name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A",
            residue_number=i + 1,
            ca_position=np.array([x, y, z]),
        ))

    return ProteinStructure(name=name, residues=residues)


def build_beta_sheet(
    n_strands: int = 4,
    strand_length: int = 8,
    name: str = "β-Sheet",
) -> ProteinStructure:
    """Build a synthetic anti-parallel β-sheet.

    Parameters
    ----------
    n_strands : int
        Number of β-strands.
    strand_length : int
        Residues per strand.
    name : str
        Structure name.

    Returns
    -------
    ProteinStructure
    """
    # β-sheet: rise 3.3Å per residue, strand spacing ~4.7Å
    rise = 3.3
    strand_spacing = 4.7
    sheet_seq = "VYIFVYIFVYIFVYIF"

    residues = []
    idx = 0

    for s in range(n_strands):
        direction = 1 if s % 2 == 0 else -1  # anti-parallel
        for r in range(strand_length):
            x = strand_spacing * s
            y = 0.0
            if direction == 1:
                z = rise * r
            else:
                z = rise * (strand_length - 1 - r)

            # Slight pleating
            y = 0.8 * ((-1) ** r)

            aa = sheet_seq[idx % len(sheet_seq)]
            residues.append(Residue(
                index=idx,
                name=aa,
                three_letter=ONE_TO_THREE.get(aa, "VAL"),
                chain_id="A",
                residue_number=idx + 1,
                ca_position=np.array([x, y, z]),
            ))
            idx += 1

    return ProteinStructure(name=name, residues=residues)


def build_helix_turn_helix(
    helix_length: int = 15,
    name: str = "Helix-Turn-Helix",
) -> ProteinStructure:
    """Build a helix-turn-helix motif.

    Two α-helices connected by a short turn, demonstrating
    both local (helical) and medium-range (turn) contacts.

    Parameters
    ----------
    helix_length : int
        Residues per helix.
    name : str
        Structure name.

    Returns
    -------
    ProteinStructure
    """
    rise = 1.5
    rpt = 3.6
    radius = 2.3
    turn_length = 4

    residues = []
    idx = 0
    motif_seq = "AELKAIAQELKAIAQGNPGAELKAIAQELKAIAQA"

    # First helix
    for i in range(helix_length):
        angle = 2 * math.pi * i / rpt
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = rise * i
        aa = motif_seq[idx % len(motif_seq)]
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A", residue_number=idx + 1,
            ca_position=np.array([x, y, z]),
        ))
        idx += 1

    # Turn region
    last_z = rise * (helix_length - 1)
    for i in range(turn_length):
        frac = (i + 1) / (turn_length + 1)
        x = radius * math.cos(math.pi) + 2 * radius * frac
        y = radius * math.sin(math.pi + math.pi * frac)
        z = last_z + rise * (i + 1) * 0.5
        aa = motif_seq[idx % len(motif_seq)]
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "GLY"),
            chain_id="A", residue_number=idx + 1,
            ca_position=np.array([x, y, z]),
        ))
        idx += 1

    # Second helix (rotated)
    offset_x = 2 * radius + 3.0
    offset_z = last_z + rise * turn_length * 0.5
    for i in range(helix_length):
        angle = 2 * math.pi * i / rpt + math.pi
        x = offset_x + radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = offset_z + rise * i
        aa = motif_seq[idx % len(motif_seq)]
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A", residue_number=idx + 1,
            ca_position=np.array([x, y, z]),
        ))
        idx += 1

    return ProteinStructure(name=name, residues=residues)


def build_beta_barrel(
    n_strands: int = 8,
    strand_length: int = 6,
    name: str = "β-Barrel",
) -> ProteinStructure:
    """Build a synthetic β-barrel (closed cylinder of β-strands).

    Parameters
    ----------
    n_strands : int
        Number of strands forming the barrel.
    strand_length : int
        Residues per strand.
    name : str
        Structure name.

    Returns
    -------
    ProteinStructure
    """
    barrel_radius = 8.0
    rise = 3.3
    barrel_seq = "VYIFWYIFVYIFWYIF"

    residues = []
    idx = 0

    for s in range(n_strands):
        angle_base = 2 * math.pi * s / n_strands
        direction = 1 if s % 2 == 0 else -1

        for r in range(strand_length):
            angle = angle_base + 0.1 * r * direction
            x = barrel_radius * math.cos(angle)
            y = barrel_radius * math.sin(angle)
            if direction == 1:
                z = rise * r
            else:
                z = rise * (strand_length - 1 - r)

            aa = barrel_seq[idx % len(barrel_seq)]
            residues.append(Residue(
                index=idx, name=aa,
                three_letter=ONE_TO_THREE.get(aa, "VAL"),
                chain_id="A", residue_number=idx + 1,
                ca_position=np.array([x, y, z]),
            ))
            idx += 1

    return ProteinStructure(name=name, residues=residues)


def build_random_coil(
    n_residues: int = 40,
    name: str = "Random Coil",
    seed: int = 42,
) -> ProteinStructure:
    """Build a random-walk polymer backbone.

    Parameters
    ----------
    n_residues : int
        Number of residues.
    name : str
        Structure name.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ProteinStructure
    """
    rng = np.random.RandomState(seed)
    bond_length = 3.8  # Cα–Cα distance

    positions = np.zeros((n_residues, 3))
    for i in range(1, n_residues):
        direction = rng.randn(3)
        direction /= np.linalg.norm(direction)
        positions[i] = positions[i - 1] + bond_length * direction

    residues = []
    all_aa = "ACDEFGHIKLMNPQRSTVWY"
    for i in range(n_residues):
        aa = all_aa[rng.randint(0, len(all_aa))]
        residues.append(Residue(
            index=i, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A", residue_number=i + 1,
            ca_position=positions[i].copy(),
        ))

    return ProteinStructure(name=name, residues=residues)


def build_two_domain(
    domain_size: int = 20,
    name: str = "Two-Domain Protein",
) -> ProteinStructure:
    """Build a two-domain protein with a linker.

    Two compact globular domains connected by a flexible linker,
    demonstrating long-range tertiary contacts within domains
    and sparse inter-domain contacts.

    Parameters
    ----------
    domain_size : int
        Residues per domain.
    name : str
        Structure name.

    Returns
    -------
    ProteinStructure
    """
    rng = np.random.RandomState(123)
    residues = []
    idx = 0
    domain_seq = "AELKFHQRSTIVMPWYDNGC"
    linker_len = 5

    # Domain 1: compact sphere-like cluster
    center1 = np.array([0.0, 0.0, 0.0])
    for i in range(domain_size):
        # Place residues on a sphere surface + some noise
        phi = math.acos(1 - 2 * (i + 0.5) / domain_size)
        theta = math.pi * (1 + 5**0.5) * i
        r = 8.0 + rng.randn() * 0.5
        pos = center1 + r * np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ])
        aa = domain_seq[idx % len(domain_seq)]
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A", residue_number=idx + 1,
            ca_position=pos,
        ))
        idx += 1

    # Linker region
    start = residues[-1].ca_position.copy()
    center2 = np.array([25.0, 0.0, 0.0])
    for i in range(linker_len):
        frac = (i + 1) / (linker_len + 1)
        pos = start * (1 - frac) + center2 * frac + rng.randn(3) * 0.5
        aa = "G" if i % 2 == 0 else "S"
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "GLY"),
            chain_id="A", residue_number=idx + 1,
            ca_position=pos,
        ))
        idx += 1

    # Domain 2: another compact cluster
    for i in range(domain_size):
        phi = math.acos(1 - 2 * (i + 0.5) / domain_size)
        theta = math.pi * (1 + 5**0.5) * i
        r = 8.0 + rng.randn() * 0.5
        pos = center2 + r * np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ])
        aa = domain_seq[idx % len(domain_seq)]
        residues.append(Residue(
            index=idx, name=aa,
            three_letter=ONE_TO_THREE.get(aa, "ALA"),
            chain_id="A", residue_number=idx + 1,
            ca_position=pos,
        ))
        idx += 1

    return ProteinStructure(name=name, residues=residues)


def get_preset_proteins() -> Dict[str, ProteinStructure]:
    """Return all preset synthetic proteins.

    Returns
    -------
    Dict[str, ProteinStructure]
        Mapping name → ProteinStructure.
    """
    return {
        "α-Helix": build_alpha_helix(),
        "β-Sheet": build_beta_sheet(),
        "Helix-Turn-Helix": build_helix_turn_helix(),
        "β-Barrel": build_beta_barrel(),
        "Random Coil": build_random_coil(),
        "Two-Domain Protein": build_two_domain(),
    }


# ═══════════════════════════════════════════════════════════════════════
# Node feature construction
# ═══════════════════════════════════════════════════════════════════════


def compute_node_features(protein: ProteinStructure) -> NodeFeatures:
    """Compute node features for each residue.

    Features per node:
        - One-hot encoding of amino acid identity (20 dims)
        - Kyte-Doolittle hydrophobicity (1 dim)
        - Net charge at pH 7 (1 dim)
        - Molecular weight (1 dim)
        - Chou-Fasman helix propensity (1 dim)
    Total: 24 features per node.

    Parameters
    ----------
    protein : ProteinStructure
        The input protein.

    Returns
    -------
    NodeFeatures
    """
    n = protein.n_residues

    # One-hot encoding
    one_hot = np.zeros((n, NUM_AMINO_ACIDS), dtype=np.float64)
    for i, res in enumerate(protein.residues):
        idx = AA_INDEX.get(res.name, 0)
        one_hot[i, idx] = 1.0

    # Scalar features
    hydro = np.array([r.hydrophobicity for r in protein.residues])
    chg = np.array([r.charge for r in protein.residues])
    wt = np.array([r.weight for r in protein.residues])
    hprop = np.array([r.helix_propensity for r in protein.residues])

    # Normalize scalar features to [0, 1]
    def _norm(arr: NDArray) -> NDArray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    hydro_norm = _norm(hydro)
    chg_norm = _norm(chg)
    wt_norm = _norm(wt)
    hprop_norm = _norm(hprop)

    # Concatenate
    feature_matrix = np.column_stack([
        one_hot,
        hydro_norm.reshape(-1, 1),
        chg_norm.reshape(-1, 1),
        wt_norm.reshape(-1, 1),
        hprop_norm.reshape(-1, 1),
    ])

    feature_names = (
        [f"AA_{aa}" for aa in AMINO_ACIDS]
        + ["hydrophobicity", "charge", "weight", "helix_propensity"]
    )

    return NodeFeatures(
        one_hot=one_hot,
        hydrophobicity=hydro,
        charge=chg,
        weight=wt,
        helix_propensity=hprop,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
    )


# ═══════════════════════════════════════════════════════════════════════
# k-NN edge construction
# ═══════════════════════════════════════════════════════════════════════


def _direction_to_quaternion(v: NDArray) -> NDArray:
    """Convert a unit direction vector to an orientation quaternion.

    The quaternion represents the rotation from the z-axis [0,0,1]
    to the direction v.

    Parameters
    ----------
    v : NDArray
        (3,) unit direction vector.

    Returns
    -------
    NDArray
        (4,) quaternion [w, x, y, z].
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    # Handle near-parallel and anti-parallel cases
    dot = np.dot(z_axis, v)

    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])

    cross = np.cross(z_axis, v)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]])
    q /= np.linalg.norm(q)
    return q


def compute_edges(
    protein: ProteinStructure,
    k: int = DEFAULT_K,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> EdgeData:
    """Build edges using k-Nearest Neighbors with a KD-Tree.

    Parameters
    ----------
    protein : ProteinStructure
        Input protein.
    k : int
        Number of nearest neighbors.
    distance_cutoff : float
        Maximum edge distance in Angstroms.

    Returns
    -------
    EdgeData
    """
    coords = protein.ca_coordinates
    n = len(coords)

    # Clamp k to at most n-1
    actual_k = min(k, n - 1)

    # Build KD-Tree
    tree = cKDTree(coords)
    distances_all, indices_all = tree.query(coords, k=actual_k + 1)
    # First column is self (distance 0), skip it
    distances_all = distances_all[:, 1:]
    indices_all = indices_all[:, 1:]

    # Build edge lists
    src_list = []
    tgt_list = []
    dist_list = []
    dir_list = []
    seq_dist_list = []
    quat_list = []

    for i in range(n):
        for j_idx in range(actual_k):
            j = indices_all[i, j_idx]
            d = distances_all[i, j_idx]

            if d > distance_cutoff:
                continue

            src_list.append(i)
            tgt_list.append(j)
            dist_list.append(d)

            # Direction vector
            r_ij = coords[j] - coords[i]
            r_norm = np.linalg.norm(r_ij)
            if r_norm > 1e-12:
                dir_vec = r_ij / r_norm
            else:
                dir_vec = np.array([0.0, 0.0, 1.0])
            dir_list.append(dir_vec)

            # Sequence distance
            seq_d = abs(i - j)
            seq_dist_list.append(seq_d)

            # Quaternion
            quat_list.append(_direction_to_quaternion(dir_vec))

    n_edges = len(src_list)

    if n_edges == 0:
        return EdgeData(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            distances=np.array([], dtype=np.float64),
            direction_vectors=np.zeros((0, 3), dtype=np.float64),
            sequence_distances=np.array([], dtype=np.int64),
            quaternions=np.zeros((0, 4), dtype=np.float64),
            n_edges=0,
        )

    edge_index = np.array([src_list, tgt_list], dtype=np.int64)
    distances = np.array(dist_list, dtype=np.float64)
    direction_vectors = np.array(dir_list, dtype=np.float64)
    sequence_distances = np.array(seq_dist_list, dtype=np.int64)
    quaternions = np.array(quat_list, dtype=np.float64)

    return EdgeData(
        edge_index=edge_index,
        distances=distances,
        direction_vectors=direction_vectors,
        sequence_distances=sequence_distances,
        quaternions=quaternions,
        n_edges=n_edges,
    )


# ═══════════════════════════════════════════════════════════════════════
# Adjacency matrix
# ═══════════════════════════════════════════════════════════════════════


def compute_adjacency(
    n_nodes: int,
    edge_data: EdgeData,
) -> AdjacencyData:
    """Build the sparse adjacency matrix from edge data.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    edge_data : EdgeData
        Edge index and features.

    Returns
    -------
    AdjacencyData
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    if edge_data.n_edges > 0:
        src = edge_data.edge_index[0]
        tgt = edge_data.edge_index[1]
        adj[src, tgt] = 1

    degree = adj.sum(axis=1).astype(np.int32)
    max_possible = n_nodes * (n_nodes - 1)
    density = float(adj.sum()) / max_possible if max_possible > 0 else 0.0

    return AdjacencyData(
        sparse_matrix=adj,
        degree=degree,
        density=density,
    )


# ═══════════════════════════════════════════════════════════════════════
# Graph statistics
# ═══════════════════════════════════════════════════════════════════════


def compute_graph_statistics(graph: ProteinGraph) -> GraphStatistics:
    """Compute summary statistics for a protein graph.

    Parameters
    ----------
    graph : ProteinGraph
        The protein graph.

    Returns
    -------
    GraphStatistics
    """
    degree = graph.adjacency.degree
    dists = graph.edge_data.distances
    seq_dists = graph.edge_data.sequence_distances

    n_short = int(np.sum(seq_dists <= 4)) if len(seq_dists) > 0 else 0
    n_medium = int(np.sum((seq_dists > 4) & (seq_dists <= 12))) if len(seq_dists) > 0 else 0
    n_long = int(np.sum(seq_dists > 12)) if len(seq_dists) > 0 else 0

    return GraphStatistics(
        n_nodes=graph.n_nodes,
        n_edges=graph.n_edges,
        density=graph.adjacency.density,
        mean_degree=float(np.mean(degree)) if len(degree) > 0 else 0.0,
        std_degree=float(np.std(degree)) if len(degree) > 0 else 0.0,
        min_degree=int(np.min(degree)) if len(degree) > 0 else 0,
        max_degree=int(np.max(degree)) if len(degree) > 0 else 0,
        mean_edge_distance=float(np.mean(dists)) if len(dists) > 0 else 0.0,
        std_edge_distance=float(np.std(dists)) if len(dists) > 0 else 0.0,
        mean_sequence_distance=float(np.mean(seq_dists)) if len(seq_dists) > 0 else 0.0,
        std_sequence_distance=float(np.std(seq_dists)) if len(seq_dists) > 0 else 0.0,
        n_short_range=n_short,
        n_medium_range=n_medium,
        n_long_range=n_long,
        feature_dim=graph.feature_dim,
        k=graph.k,
        distance_cutoff=graph.distance_cutoff,
    )


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════


def protein_to_graph(
    protein: ProteinStructure,
    k: int = DEFAULT_K,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> ProteinGraph:
    """Full pipeline: convert a protein to a graph.

    Parameters
    ----------
    protein : ProteinStructure
        Input protein.
    k : int
        Number of nearest neighbors.
    distance_cutoff : float
        Maximum edge distance (Å).

    Returns
    -------
    ProteinGraph
    """
    node_features = compute_node_features(protein)
    edge_data = compute_edges(protein, k=k, distance_cutoff=distance_cutoff)
    adjacency = compute_adjacency(protein.n_residues, edge_data)

    return ProteinGraph(
        protein=protein,
        node_features=node_features,
        edge_data=edge_data,
        adjacency=adjacency,
        k=k,
        distance_cutoff=distance_cutoff,
    )


def build_graph_from_pdb(
    pdb_text: str,
    k: int = DEFAULT_K,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
    name: str = "protein",
) -> ProteinGraph:
    """One-shot: PDB text → ProteinGraph.

    Parameters
    ----------
    pdb_text : str
        PDB file contents.
    k : int
        KNN neighbors.
    distance_cutoff : float
        Edge cutoff in Å.
    name : str
        Protein name.

    Returns
    -------
    ProteinGraph
    """
    protein = parse_pdb(pdb_text, name=name)
    return protein_to_graph(protein, k=k, distance_cutoff=distance_cutoff)


def sweep_k(
    protein: ProteinStructure,
    k_values: Optional[List[int]] = None,
    distance_cutoff: float = DEFAULT_DISTANCE_CUTOFF,
) -> List[ProteinGraph]:
    """Build graphs at multiple k values.

    Parameters
    ----------
    protein : ProteinStructure
        Input protein.
    k_values : List[int], optional
        k values to sweep. Defaults to [2, 4, 6, 8, 10, 15, 20].
    distance_cutoff : float
        Edge cutoff in Å.

    Returns
    -------
    List[ProteinGraph]
    """
    if k_values is None:
        k_values = [2, 4, 6, 8, 10, 15, 20]

    return [
        protein_to_graph(protein, k=k, distance_cutoff=distance_cutoff)
        for k in k_values
    ]


# ═══════════════════════════════════════════════════════════════════════
# Contact classification helpers
# ═══════════════════════════════════════════════════════════════════════


def classify_contact(seq_dist: int) -> str:
    """Classify a contact by sequence distance.

    Parameters
    ----------
    seq_dist : int
        |i - j| sequence separation.

    Returns
    -------
    str
        Contact type: 'backbone', 'short-range', 'medium-range', or 'long-range'.
    """
    if seq_dist <= 1:
        return "backbone"
    elif seq_dist <= 4:
        return "short-range"
    elif seq_dist <= 12:
        return "medium-range"
    else:
        return "long-range"


def classify_contacts(edge_data: EdgeData) -> List[str]:
    """Classify all edges by sequence distance.

    Parameters
    ----------
    edge_data : EdgeData
        The edge data.

    Returns
    -------
    List[str]
        Contact type for each edge.
    """
    return [classify_contact(int(d)) for d in edge_data.sequence_distances]


CONTACT_COLORS: Dict[str, str] = {
    "backbone": "#888888",
    "short-range": "#2196F3",    # Blue — i to i+2..4 (α-helix)
    "medium-range": "#FF9800",   # Orange — i to i+5..12
    "long-range": "#F44336",     # Red — i to i+13+ (tertiary)
}

CONTACT_DESCRIPTIONS: Dict[str, str] = {
    "backbone": "Sequential neighbors (i±1)",
    "short-range": "α-helix contacts (i±2 to i±4)",
    "medium-range": "Turns and loops (i±5 to i±12)",
    "long-range": "Tertiary contacts (i±13+), the hard part of folding",
}
