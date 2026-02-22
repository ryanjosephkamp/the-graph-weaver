#!/usr/bin/env python3
"""
The Graph Weaver — CLI Entry Point.

Four analysis modes:
    --analyze        Standard graph analysis of a single protein
    --compare        Compare all six preset proteins
    --sweep          k-sweep: vary k and track graph properties
    --contacts       Detailed contact-type analysis

Usage:
    python main.py                                  # default: α-Helix analysis
    python main.py --analyze --protein helix --save
    python main.py --compare --save
    python main.py --sweep --protein barrel --save
    python main.py --contacts --protein two_domain --save --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

from src.graph_engine import (
    ProteinStructure,
    build_alpha_helix,
    build_beta_sheet,
    build_helix_turn_helix,
    build_beta_barrel,
    build_random_coil,
    build_two_domain,
    DEFAULT_K,
    DEFAULT_DISTANCE_CUTOFF,
)
from src.analysis import (
    analyze_graph,
    analyze_k_sweep,
    analyze_contacts,
    analyze_features,
    compare_preset_proteins,
    graph_summary,
)
from src.visualization import MatplotlibRenderer


# ── Protein presets ──
PROTEIN_MAP = {
    "helix":    ("α-Helix",            build_alpha_helix),
    "sheet":    ("β-Sheet",            build_beta_sheet),
    "hth":      ("Helix-Turn-Helix",   build_helix_turn_helix),
    "barrel":   ("β-Barrel",           build_beta_barrel),
    "coil":     ("Random Coil",        build_random_coil),
    "two_domain": ("Two-Domain Protein", build_two_domain),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="graph_weaver",
        description="The Graph Weaver — Protein-to-Graph Conversion Pipeline",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", default=True,
                      help="Standard graph analysis (default)")
    mode.add_argument("--compare", action="store_true",
                      help="Compare all six preset proteins")
    mode.add_argument("--sweep", action="store_true",
                      help="k-sweep analysis")
    mode.add_argument("--contacts", action="store_true",
                      help="Detailed contact-type analysis")

    parser.add_argument("--protein", type=str, default="helix",
                        choices=list(PROTEIN_MAP.keys()),
                        help="Preset protein to analyse (default: helix)")
    parser.add_argument("-k", "--k-neighbors", type=int, default=DEFAULT_K,
                        help=f"k-NN neighbors (default: {DEFAULT_K})")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_DISTANCE_CUTOFF,
                        help=f"Distance cutoff in Å (default: {DEFAULT_DISTANCE_CUTOFF})")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to figures/")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    return parser


def _get_protein(name: str) -> ProteinStructure:
    """Look up and build a preset protein."""
    display_name, builder = PROTEIN_MAP[name]
    return builder()


def _ensure_figures():
    """Create figures/ directory if needed."""
    os.makedirs("figures", exist_ok=True)


def _save_fig(fig, name: str):
    """Save a Matplotlib figure."""
    _ensure_figures()
    path = os.path.join("figures", f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# ── Commands ──


def cmd_analyze(args):
    """Standard graph analysis."""
    t0 = time.time()
    protein = _get_protein(args.protein)
    k = args.k_neighbors
    cutoff = args.cutoff

    print(f"\n{'═' * 60}")
    print(f"  Graph Analysis: {protein.name}")
    print(f"  k = {k}, cutoff = {cutoff:.1f} Å")
    print(f"{'═' * 60}\n")

    analysis = analyze_graph(protein, k=k, distance_cutoff=cutoff)
    print(graph_summary(analysis))

    if args.verbose:
        print(f"\n{analysis.explanation}")

    if args.save:
        fig1 = MatplotlibRenderer.graph_2d(
            analysis.graph,
            title=f"Protein Graph — {protein.name}",
        )
        _save_fig(fig1, f"graph_2d_{args.protein}")

        fig2 = MatplotlibRenderer.adjacency_matrix(
            analysis.graph,
            title=f"Adjacency Matrix — {protein.name}",
        )
        _save_fig(fig2, f"adjacency_{args.protein}")

        fig3 = MatplotlibRenderer.contact_type_bar(
            analysis,
            title=f"Contact Types — {protein.name}",
        )
        _save_fig(fig3, f"contacts_{args.protein}")

        fig4 = MatplotlibRenderer.degree_histogram(
            analysis,
            title=f"Degree Distribution — {protein.name}",
        )
        _save_fig(fig4, f"degree_{args.protein}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.2f}s")


def cmd_compare(args):
    """Compare all preset proteins."""
    t0 = time.time()
    k = args.k_neighbors
    cutoff = args.cutoff

    print(f"\n{'═' * 60}")
    print(f"  Preset Comparison (k={k}, cutoff={cutoff:.1f}Å)")
    print(f"{'═' * 60}\n")

    comparison = compare_preset_proteins(k=k, distance_cutoff=cutoff)

    # Print summary table
    header = f"{'Protein':25s} {'Residues':>8s} {'Edges':>8s} {'Density':>10s} {'Mean Deg':>10s} {'LR %':>8s}"
    print(header)
    print("─" * len(header))
    for row in comparison.summary_table:
        print(f"{row['Protein']:25s} {row['Residues']:>8d} {row['Edges']:>8d} "
              f"{row['Density']:>10s} {row['Mean Degree']:>10s} {row['Long-Range %']:>8s}")

    if args.save:
        fig = MatplotlibRenderer.preset_comparison(comparison)
        _save_fig(fig, "preset_comparison")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.2f}s")


def cmd_sweep(args):
    """k-sweep analysis."""
    t0 = time.time()
    protein = _get_protein(args.protein)
    cutoff = args.cutoff

    print(f"\n{'═' * 60}")
    print(f"  k-Sweep: {protein.name} (cutoff={cutoff:.1f}Å)")
    print(f"{'═' * 60}\n")

    sweep = analyze_k_sweep(protein, distance_cutoff=cutoff)
    print(sweep.explanation)

    # Table
    header = f"{'k':>5s} {'Edges':>8s} {'Density':>10s} {'Mean Deg':>10s} {'LR %':>8s}"
    print(header)
    print("─" * len(header))
    for i, k in enumerate(sweep.k_values):
        s = sweep.statistics[i]
        lr = sweep.long_range_fractions[i] * 100
        print(f"{k:>5d} {s.n_edges:>8d} {s.density:>10.4f} "
              f"{s.mean_degree:>10.1f} {lr:>8.1f}")

    if args.save:
        fig = MatplotlibRenderer.k_sweep_summary(sweep)
        _save_fig(fig, f"k_sweep_{args.protein}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.2f}s")


def cmd_contacts(args):
    """Detailed contact analysis."""
    t0 = time.time()
    protein = _get_protein(args.protein)
    k = args.k_neighbors
    cutoff = args.cutoff

    print(f"\n{'═' * 60}")
    print(f"  Contact Analysis: {protein.name}")
    print(f"  k = {k}, cutoff = {cutoff:.1f} Å")
    print(f"{'═' * 60}\n")

    from src.graph_engine import protein_to_graph
    graph = protein_to_graph(protein, k=k, distance_cutoff=cutoff)
    contact_analysis = analyze_contacts(graph)
    print(contact_analysis.explanation)

    if args.verbose:
        feature_analysis = analyze_features(graph)
        print(feature_analysis.explanation)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.2f}s")


def main():
    """Main dispatch."""
    parser = build_parser()
    args = parser.parse_args()

    if args.compare:
        cmd_compare(args)
    elif args.sweep:
        cmd_sweep(args)
    elif args.contacts:
        cmd_contacts(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
