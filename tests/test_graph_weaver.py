"""
The Graph Weaver — Test Suite.

18 test classes, 90+ test methods covering all modules:
    - Engine: Constants, Residue, ProteinStructure, builders, node features,
      edge construction, adjacency, graph statistics, contacts, pipeline
    - Analysis: FullGraphAnalysis, KSweep, Contacts, Features, Comparison, Summary
    - Visualization: PlotlyRenderer, MatplotlibRenderer
    - CLI: argument parsing
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    # Pipeline
    compute_node_features,
    compute_edges,
    compute_adjacency,
    compute_graph_statistics,
    protein_to_graph,
    build_graph_from_pdb,
    sweep_k,
    # Contacts
    classify_contact,
    classify_contacts,
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
# Test Constants
# ═══════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test global constants and lookup tables."""

    def test_amino_acid_count(self):
        assert len(AMINO_ACIDS) == 20

    def test_amino_acid_index(self):
        assert len(AA_INDEX) == 20
        assert AA_INDEX["A"] == 0
        assert AA_INDEX["Y"] == 19

    def test_num_amino_acids(self):
        assert NUM_AMINO_ACIDS == 20

    def test_three_to_one_mapping(self):
        assert len(THREE_TO_ONE) == 20
        assert THREE_TO_ONE["ALA"] == "A"
        assert THREE_TO_ONE["TYR"] == "Y"

    def test_one_to_three_mapping(self):
        assert len(ONE_TO_THREE) == 20
        assert ONE_TO_THREE["A"] == "ALA"

    def test_hydrophobicity_keys(self):
        assert set(HYDROPHOBICITY.keys()) == set(AMINO_ACIDS)

    def test_charge_keys(self):
        assert set(CHARGE.keys()) == set(AMINO_ACIDS)

    def test_molecular_weight_keys(self):
        assert set(MOLECULAR_WEIGHT.keys()) == set(AMINO_ACIDS)

    def test_helix_propensity_keys(self):
        assert set(HELIX_PROPENSITY.keys()) == set(AMINO_ACIDS)

    def test_default_k(self):
        assert DEFAULT_K == 10

    def test_default_cutoff(self):
        assert DEFAULT_DISTANCE_CUTOFF == 10.0

    def test_contact_colors(self):
        assert "backbone" in CONTACT_COLORS
        assert "short-range" in CONTACT_COLORS
        assert "medium-range" in CONTACT_COLORS
        assert "long-range" in CONTACT_COLORS

    def test_contact_descriptions(self):
        assert len(CONTACT_DESCRIPTIONS) == 4


# ═══════════════════════════════════════════════════════════════════════
# Test Residue
# ═══════════════════════════════════════════════════════════════════════


class TestResidue:
    """Test the Residue dataclass."""

    def test_create_residue(self):
        r = Residue(
            index=0, name="A", three_letter="ALA",
            chain_id="A", residue_number=1,
            ca_position=np.array([1.0, 2.0, 3.0]),
        )
        assert r.index == 0
        assert r.name == "A"

    def test_position_property(self):
        r = Residue(
            index=0, name="A", three_letter="ALA",
            chain_id="A", residue_number=1,
            ca_position=np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(r.position, r.ca_position)

    def test_hydrophobicity_property(self):
        r = Residue(
            index=0, name="I", three_letter="ILE",
            chain_id="A", residue_number=1,
            ca_position=np.zeros(3),
        )
        assert r.hydrophobicity == 4.5

    def test_charge_property(self):
        r = Residue(
            index=0, name="K", three_letter="LYS",
            chain_id="A", residue_number=1,
            ca_position=np.zeros(3),
        )
        assert r.charge == 1.0

    def test_weight_property(self):
        r = Residue(
            index=0, name="G", three_letter="GLY",
            chain_id="A", residue_number=1,
            ca_position=np.zeros(3),
        )
        assert r.weight == 75.0

    def test_helix_propensity_property(self):
        r = Residue(
            index=0, name="A", three_letter="ALA",
            chain_id="A", residue_number=1,
            ca_position=np.zeros(3),
        )
        assert r.helix_propensity == 1.42


# ═══════════════════════════════════════════════════════════════════════
# Test ProteinStructure
# ═══════════════════════════════════════════════════════════════════════


class TestProteinStructure:
    """Test the ProteinStructure dataclass."""

    @pytest.fixture
    def simple_protein(self):
        residues = [
            Residue(i, AMINO_ACIDS[i % 20], list(THREE_TO_ONE.keys())[i % 20],
                    "A", i + 1, np.array([float(i), 0.0, 0.0]))
            for i in range(5)
        ]
        return ProteinStructure(name="test", residues=residues)

    def test_n_residues(self, simple_protein):
        assert simple_protein.n_residues == 5

    def test_sequence(self, simple_protein):
        assert len(simple_protein.sequence) == 5

    def test_ca_coordinates_shape(self, simple_protein):
        coords = simple_protein.ca_coordinates
        assert coords.shape == (5, 3)

    def test_chain_ids(self, simple_protein):
        assert "A" in simple_protein.chain_ids


# ═══════════════════════════════════════════════════════════════════════
# Test Builders
# ═══════════════════════════════════════════════════════════════════════


class TestBuilders:
    """Test synthetic protein builders."""

    def test_alpha_helix_default(self):
        protein = build_alpha_helix()
        assert protein.n_residues == 30
        assert protein.name == "α-Helix"

    def test_alpha_helix_custom(self):
        protein = build_alpha_helix(n_residues=10, name="test")
        assert protein.n_residues == 10
        assert protein.name == "test"

    def test_beta_sheet_default(self):
        protein = build_beta_sheet()
        assert protein.n_residues == 32  # 4 strands × 8 residues
        assert protein.name == "β-Sheet"

    def test_helix_turn_helix_default(self):
        protein = build_helix_turn_helix()
        assert protein.n_residues == 34  # 15 + 4 + 15
        assert protein.name == "Helix-Turn-Helix"

    def test_beta_barrel_default(self):
        protein = build_beta_barrel()
        assert protein.n_residues == 48  # 8 strands × 6 residues
        assert protein.name == "β-Barrel"

    def test_random_coil_default(self):
        protein = build_random_coil()
        assert protein.n_residues == 40
        assert protein.name == "Random Coil"

    def test_random_coil_reproducible(self):
        p1 = build_random_coil(seed=42)
        p2 = build_random_coil(seed=42)
        np.testing.assert_array_equal(
            p1.ca_coordinates, p2.ca_coordinates
        )

    def test_two_domain_default(self):
        protein = build_two_domain()
        assert protein.n_residues == 45  # 20 + 5 + 20
        assert protein.name == "Two-Domain Protein"

    def test_get_preset_proteins(self):
        presets = get_preset_proteins()
        assert len(presets) == 6
        assert "α-Helix" in presets
        assert "β-Barrel" in presets

    def test_all_presets_have_residues(self):
        for name, protein in get_preset_proteins().items():
            assert protein.n_residues > 0, f"{name} has no residues"

    def test_all_presets_have_coordinates(self):
        for name, protein in get_preset_proteins().items():
            coords = protein.ca_coordinates
            assert coords.shape[0] == protein.n_residues
            assert coords.shape[1] == 3

    def test_helix_coordinates_form_helix(self):
        protein = build_alpha_helix(n_residues=10)
        coords = protein.ca_coordinates
        # Check z increases monotonically
        z_vals = coords[:, 2]
        assert np.all(np.diff(z_vals) > 0)


# ═══════════════════════════════════════════════════════════════════════
# Test PDB Parsing
# ═══════════════════════════════════════════════════════════════════════


class TestPDBParsing:
    """Test PDB text parsing."""

    @pytest.fixture
    def sample_pdb(self):
        return """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       5.000   6.000   7.000  1.00  0.00           C
ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00  0.00           C
ATOM      7  N   VAL A   3       7.000   8.000   9.000  1.00  0.00           N
ATOM      8  CA  VAL A   3       8.000   9.000  10.000  1.00  0.00           C
END
"""

    def test_parse_pdb_extracts_ca(self, sample_pdb):
        protein = parse_pdb(sample_pdb, name="test")
        assert protein.n_residues == 3

    def test_parse_pdb_residue_names(self, sample_pdb):
        protein = parse_pdb(sample_pdb, name="test")
        assert protein.residues[0].name == "A"
        assert protein.residues[1].name == "G"
        assert protein.residues[2].name == "V"

    def test_parse_pdb_coordinates(self, sample_pdb):
        protein = parse_pdb(sample_pdb, name="test")
        np.testing.assert_array_almost_equal(
            protein.residues[0].ca_position, [2.0, 3.0, 4.0]
        )

    def test_parse_pdb_no_duplicates(self, sample_pdb):
        # Add a duplicate CA for residue 1
        duplicate_pdb = sample_pdb + "ATOM      9  CA  ALA A   1       9.000  10.000  11.000  1.00  0.00           C\n"
        protein = parse_pdb(duplicate_pdb, name="test")
        assert protein.n_residues == 3  # Should not add duplicate


# ═══════════════════════════════════════════════════════════════════════
# Test Node Features
# ═══════════════════════════════════════════════════════════════════════


class TestNodeFeatures:
    """Test node feature computation."""

    @pytest.fixture
    def helix_protein(self):
        return build_alpha_helix(n_residues=10)

    def test_one_hot_shape(self, helix_protein):
        features = compute_node_features(helix_protein)
        assert features.one_hot.shape == (10, 20)

    def test_one_hot_sums_to_one(self, helix_protein):
        features = compute_node_features(helix_protein)
        sums = features.one_hot.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(10))

    def test_feature_matrix_shape(self, helix_protein):
        features = compute_node_features(helix_protein)
        assert features.feature_matrix.shape == (10, 24)

    def test_feature_names_count(self, helix_protein):
        features = compute_node_features(helix_protein)
        assert len(features.feature_names) == 24

    def test_hydrophobicity_array(self, helix_protein):
        features = compute_node_features(helix_protein)
        assert len(features.hydrophobicity) == 10

    def test_charge_array(self, helix_protein):
        features = compute_node_features(helix_protein)
        assert len(features.charge) == 10


# ═══════════════════════════════════════════════════════════════════════
# Test Edge Construction
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeConstruction:
    """Test k-NN edge construction."""

    @pytest.fixture
    def helix_protein(self):
        return build_alpha_helix(n_residues=10)

    def test_edge_index_shape(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        assert edges.edge_index.shape[0] == 2

    def test_edge_count_positive(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        assert edges.n_edges > 0

    def test_distances_positive(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        assert np.all(edges.distances > 0)

    def test_direction_vectors_unit(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        norms = np.linalg.norm(edges.direction_vectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(edges.n_edges))

    def test_sequence_distances_nonnegative(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        assert np.all(edges.sequence_distances >= 0)

    def test_quaternions_shape(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        assert edges.quaternions.shape == (edges.n_edges, 4)

    def test_quaternions_unit(self, helix_protein):
        edges = compute_edges(helix_protein, k=5)
        norms = np.linalg.norm(edges.quaternions, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(edges.n_edges))

    def test_distance_cutoff(self, helix_protein):
        edges = compute_edges(helix_protein, k=5, distance_cutoff=5.0)
        assert np.all(edges.distances <= 5.0)

    def test_k_clamp(self):
        """k is clamped to n-1 for small proteins."""
        protein = build_alpha_helix(n_residues=3)
        edges = compute_edges(protein, k=100)
        # Should not crash; k is clamped
        assert edges.n_edges > 0

    def test_more_edges_with_higher_k(self, helix_protein):
        edges_low = compute_edges(helix_protein, k=3, distance_cutoff=50.0)
        edges_high = compute_edges(helix_protein, k=8, distance_cutoff=50.0)
        assert edges_high.n_edges >= edges_low.n_edges


# ═══════════════════════════════════════════════════════════════════════
# Test Adjacency
# ═══════════════════════════════════════════════════════════════════════


class TestAdjacency:
    """Test adjacency matrix construction."""

    def test_adjacency_shape(self):
        protein = build_alpha_helix(n_residues=10)
        edges = compute_edges(protein, k=5)
        adj = compute_adjacency(10, edges)
        assert adj.sparse_matrix.shape == (10, 10)

    def test_adjacency_binary(self):
        protein = build_alpha_helix(n_residues=10)
        edges = compute_edges(protein, k=5)
        adj = compute_adjacency(10, edges)
        assert set(np.unique(adj.sparse_matrix)).issubset({0, 1})

    def test_degree_sum(self):
        protein = build_alpha_helix(n_residues=10)
        edges = compute_edges(protein, k=5)
        adj = compute_adjacency(10, edges)
        assert adj.degree.sum() == adj.sparse_matrix.sum()

    def test_density_range(self):
        protein = build_alpha_helix(n_residues=10)
        edges = compute_edges(protein, k=5)
        adj = compute_adjacency(10, edges)
        assert 0.0 <= adj.density <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Test Graph Statistics
# ═══════════════════════════════════════════════════════════════════════


class TestGraphStatistics:
    """Test graph statistics computation."""

    @pytest.fixture
    def helix_graph(self):
        protein = build_alpha_helix(n_residues=20)
        return protein_to_graph(protein, k=8)

    def test_n_nodes(self, helix_graph):
        stats = compute_graph_statistics(helix_graph)
        assert stats.n_nodes == 20

    def test_n_edges(self, helix_graph):
        stats = compute_graph_statistics(helix_graph)
        assert stats.n_edges > 0

    def test_density(self, helix_graph):
        stats = compute_graph_statistics(helix_graph)
        assert 0.0 < stats.density < 1.0

    def test_degree_stats(self, helix_graph):
        stats = compute_graph_statistics(helix_graph)
        assert stats.mean_degree > 0
        assert stats.min_degree >= 0
        assert stats.max_degree >= stats.min_degree

    def test_contact_counts(self, helix_graph):
        stats = compute_graph_statistics(helix_graph)
        total = stats.n_short_range + stats.n_medium_range + stats.n_long_range
        # backbone counts are included in short-range
        assert total <= stats.n_edges


# ═══════════════════════════════════════════════════════════════════════
# Test Contact Classification
# ═══════════════════════════════════════════════════════════════════════


class TestContactClassification:
    """Test contact type classification."""

    def test_backbone_contact(self):
        assert classify_contact(1) == "backbone"

    def test_short_range_contact(self):
        assert classify_contact(3) == "short-range"
        assert classify_contact(4) == "short-range"

    def test_medium_range_contact(self):
        assert classify_contact(5) == "medium-range"
        assert classify_contact(12) == "medium-range"

    def test_long_range_contact(self):
        assert classify_contact(13) == "long-range"
        assert classify_contact(50) == "long-range"

    def test_classify_contacts_list(self):
        protein = build_alpha_helix(n_residues=20)
        edges = compute_edges(protein, k=8)
        types = classify_contacts(edges)
        assert len(types) == edges.n_edges
        assert all(t in ["backbone", "short-range", "medium-range", "long-range"]
                    for t in types)


# ═══════════════════════════════════════════════════════════════════════
# Test Full Pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """Test the full protein-to-graph pipeline."""

    def test_protein_to_graph(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        assert isinstance(graph, ProteinGraph)
        assert graph.n_nodes == 20
        assert graph.n_edges > 0
        assert graph.feature_dim == 24

    def test_build_graph_from_pdb(self):
        pdb_text = "\n".join([
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C",
            "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00           C",
            "ATOM      3  CA  VAL A   3       7.600   0.000   0.000  1.00  0.00           C",
        ])
        graph = build_graph_from_pdb(pdb_text, name="test")
        assert graph.n_nodes == 3
        assert graph.n_edges > 0

    def test_sweep_k(self):
        protein = build_alpha_helix(n_residues=10)
        graphs = sweep_k(protein, k_values=[2, 5, 8])
        assert len(graphs) == 3
        assert graphs[0].k == 2
        assert graphs[2].k == 8


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — FullGraphAnalysis
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeGraph:
    """Test the analyze_graph pipeline."""

    def test_analyze_returns_analysis(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        assert isinstance(analysis, FullGraphAnalysis)

    def test_analysis_has_graph(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        assert analysis.graph is not None
        assert analysis.graph.n_nodes == 20

    def test_analysis_has_statistics(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        assert analysis.statistics.n_nodes == 20

    def test_contact_fractions_sum(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        total = sum(analysis.contact_fractions.values())
        assert abs(total - 1.0) < 0.01

    def test_aa_composition(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        total_aa = sum(analysis.aa_composition.values())
        assert total_aa == 20

    def test_explanation_not_empty(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        assert len(analysis.explanation) > 0


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — KSweep
# ═══════════════════════════════════════════════════════════════════════


class TestKSweep:
    """Test the k-sweep analysis."""

    def test_sweep_returns_analysis(self):
        protein = build_alpha_helix(n_residues=15)
        sweep = analyze_k_sweep(protein, k_values=[2, 5, 8])
        assert isinstance(sweep, KSweepAnalysis)

    def test_sweep_correct_k_values(self):
        protein = build_alpha_helix(n_residues=15)
        sweep = analyze_k_sweep(protein, k_values=[2, 5, 8])
        assert sweep.k_values == [2, 5, 8]

    def test_sweep_edges_increase(self):
        protein = build_alpha_helix(n_residues=15)
        sweep = analyze_k_sweep(protein, k_values=[2, 5, 8],
                                distance_cutoff=50.0)
        # Edge count should generally increase with k
        assert sweep.edge_counts[-1] >= sweep.edge_counts[0]

    def test_sweep_graphs_count(self):
        protein = build_alpha_helix(n_residues=15)
        sweep = analyze_k_sweep(protein, k_values=[2, 5, 8])
        assert len(sweep.graphs) == 3


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — Contacts
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeContacts:
    """Test contact analysis."""

    def test_contact_analysis(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        ca = analyze_contacts(graph)
        assert isinstance(ca, ContactAnalysis)

    def test_contact_counts(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        ca = analyze_contacts(graph)
        total = sum(ca.contact_counts.values())
        assert total == graph.n_edges

    def test_contact_edge_subsets(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        ca = analyze_contacts(graph)
        # Sum of subset sizes should equal total
        subset_total = (
            ca.backbone_edges.shape[1]
            + ca.short_range_edges.shape[1]
            + ca.medium_range_edges.shape[1]
            + ca.long_range_edges.shape[1]
        )
        assert subset_total == graph.n_edges


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — Features
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeFeatures:
    """Test feature analysis."""

    def test_feature_analysis(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        fa = analyze_features(graph)
        assert isinstance(fa, FeatureAnalysis)

    def test_feature_matrix_shape(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        fa = analyze_features(graph)
        assert fa.feature_matrix.shape == (20, 24)

    def test_feature_means_shape(self):
        protein = build_alpha_helix(n_residues=20)
        graph = protein_to_graph(protein, k=8)
        fa = analyze_features(graph)
        assert fa.feature_means.shape == (24,)


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — Preset Comparison
# ═══════════════════════════════════════════════════════════════════════


class TestPresetComparison:
    """Test preset protein comparison."""

    def test_comparison_result(self):
        comparison = compare_preset_proteins(k=5, distance_cutoff=15.0)
        assert isinstance(comparison, PresetComparisonResult)

    def test_comparison_has_all_presets(self):
        comparison = compare_preset_proteins(k=5, distance_cutoff=15.0)
        assert len(comparison.analyses) == 6

    def test_summary_table(self):
        comparison = compare_preset_proteins(k=5, distance_cutoff=15.0)
        assert len(comparison.summary_table) == 6
        assert "Protein" in comparison.summary_table[0]


# ═══════════════════════════════════════════════════════════════════════
# Test Analysis — Summary
# ═══════════════════════════════════════════════════════════════════════


class TestGraphSummary:
    """Test human-readable summary generation."""

    def test_summary_not_empty(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        summary = graph_summary(analysis)
        assert len(summary) > 0

    def test_summary_contains_name(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        summary = graph_summary(analysis)
        assert "α-Helix" in summary

    def test_summary_contains_metrics(self):
        protein = build_alpha_helix(n_residues=20)
        analysis = analyze_graph(protein, k=8)
        summary = graph_summary(analysis)
        assert "Residues" in summary
        assert "Edges" in summary
        assert "Density" in summary


# ═══════════════════════════════════════════════════════════════════════
# Test Visualization — PlotlyRenderer
# ═══════════════════════════════════════════════════════════════════════


class TestPlotlyRenderer:
    """Test Plotly visualization methods."""

    @pytest.fixture
    def analysis(self):
        protein = build_alpha_helix(n_residues=15)
        return analyze_graph(protein, k=5)

    @pytest.fixture
    def sweep(self):
        protein = build_alpha_helix(n_residues=15)
        return analyze_k_sweep(protein, k_values=[2, 5, 8])

    def test_graph_3d(self, analysis):
        fig = PlotlyRenderer.graph_3d(analysis.graph)
        assert fig is not None

    def test_adjacency_heatmap(self, analysis):
        fig = PlotlyRenderer.adjacency_heatmap(analysis.graph)
        assert fig is not None

    def test_contact_map(self, analysis):
        fig = PlotlyRenderer.contact_map(analysis.graph)
        assert fig is not None

    def test_degree_histogram(self, analysis):
        fig = PlotlyRenderer.degree_histogram(analysis)
        assert fig is not None

    def test_distance_histogram(self, analysis):
        fig = PlotlyRenderer.distance_histogram(analysis)
        assert fig is not None

    def test_sequence_distance_histogram(self, analysis):
        fig = PlotlyRenderer.sequence_distance_histogram(analysis)
        assert fig is not None

    def test_contact_type_pie(self, analysis):
        fig = PlotlyRenderer.contact_type_pie(analysis)
        assert fig is not None

    def test_k_sweep_edges(self, sweep):
        fig = PlotlyRenderer.k_sweep_edges(sweep)
        assert fig is not None

    def test_k_sweep_density(self, sweep):
        fig = PlotlyRenderer.k_sweep_density(sweep)
        assert fig is not None

    def test_k_sweep_long_range(self, sweep):
        fig = PlotlyRenderer.k_sweep_long_range(sweep)
        assert fig is not None

    def test_feature_heatmap(self, analysis):
        fa = analyze_features(analysis.graph)
        fig = PlotlyRenderer.feature_heatmap(fa)
        assert fig is not None

    def test_hydrophobicity_profile(self, analysis):
        fa = analyze_features(analysis.graph)
        fig = PlotlyRenderer.hydrophobicity_profile(fa)
        assert fig is not None

    def test_preset_comparison_bars(self):
        comparison = compare_preset_proteins(k=5, distance_cutoff=15.0)
        fig = PlotlyRenderer.preset_comparison_bars(comparison)
        assert fig is not None


# ═══════════════════════════════════════════════════════════════════════
# Test Visualization — MatplotlibRenderer
# ═══════════════════════════════════════════════════════════════════════


class TestMatplotlibRenderer:
    """Test Matplotlib visualization methods."""

    @pytest.fixture
    def analysis(self):
        protein = build_alpha_helix(n_residues=15)
        return analyze_graph(protein, k=5)

    def test_graph_2d(self, analysis):
        fig = MatplotlibRenderer.graph_2d(analysis.graph)
        assert fig is not None
        plt_close(fig)

    def test_adjacency_matrix(self, analysis):
        fig = MatplotlibRenderer.adjacency_matrix(analysis.graph)
        assert fig is not None
        plt_close(fig)

    def test_contact_type_bar(self, analysis):
        fig = MatplotlibRenderer.contact_type_bar(analysis)
        assert fig is not None
        plt_close(fig)

    def test_degree_histogram(self, analysis):
        fig = MatplotlibRenderer.degree_histogram(analysis)
        assert fig is not None
        plt_close(fig)

    def test_k_sweep_summary(self):
        protein = build_alpha_helix(n_residues=15)
        sweep = analyze_k_sweep(protein, k_values=[2, 5, 8])
        fig = MatplotlibRenderer.k_sweep_summary(sweep)
        assert fig is not None
        plt_close(fig)

    def test_preset_comparison(self):
        comparison = compare_preset_proteins(k=5, distance_cutoff=15.0)
        fig = MatplotlibRenderer.preset_comparison(comparison)
        assert fig is not None
        plt_close(fig)


def plt_close(fig):
    """Close a Matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Test CLI
# ═══════════════════════════════════════════════════════════════════════


class TestCLI:
    """Test CLI argument parsing."""

    def test_default_args(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        assert args.protein == "helix"
        assert args.k_neighbors == DEFAULT_K

    def test_analyze_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--analyze", "--protein", "barrel"])
        assert args.protein == "barrel"

    def test_compare_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--compare"])
        assert args.compare is True

    def test_sweep_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--sweep", "--protein", "sheet"])
        assert args.sweep is True
        assert args.protein == "sheet"

    def test_contacts_mode(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--contacts"])
        assert args.contacts is True

    def test_save_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--save"])
        assert args.save is True

    def test_verbose_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_k_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["-k", "15"])
        assert args.k_neighbors == 15

    def test_cutoff_flag(self):
        from main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cutoff", "8.0"])
        assert args.cutoff == 8.0
