#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc Evaluation for HyDRA
=============================
Computes multi-scale hypergraph retrieval quality and pseudo-label precision
from pipeline output files. No pipeline changes required.

Usage:
    python evaluate_metrics.py --data_root /path/to/data/icews_wiki --task all
    python evaluate_metrics.py --data_root /path/to/data/1m/en_de   --task retrieval_quality
    python evaluate_metrics.py --data_root /path/to/data/1m/en_de   --task pseudo_precision

Supports flat (e.g., icews_wiki) and partitioned (e.g., 1m/en_de) layouts.
"""

import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Data loading helpers
# ============================================================================

def load_ref_pairs(data_root: str) -> Dict[int, int]:
    """Load ground-truth ref_pairs. Supports flat and partitioned layouts."""
    for path in [
        os.path.join(data_root, "partition_1", "ref_pairs"),
        os.path.join(data_root, "ref_pairs"),
    ]:
        if os.path.exists(path):
            pairs = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            pairs[int(parts[0])] = int(parts[1])
                        except ValueError:
                            continue
            return pairs
    return {}


def load_candidates(filepath: str) -> Dict[int, List[int]]:
    """Load a candidate pool: {kg1_id: [kg2_ids_ordered_by_rank]}."""
    cands = defaultdict(list)
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    kg1, kg2 = int(parts[0]), int(parts[1])
                    if kg2 not in cands[kg1]:
                        cands[kg1].append(kg2)
                except ValueError:
                    continue
    return dict(cands)


def _resolve_dirs(data_root: str) -> Tuple[str, str, bool]:
    """
    Resolve mp_dir, hg_dir, and is_partitioned from data_root.
    Supports both flat (e.g., icews_wiki) and partitioned (e.g., 1m/en_de) layouts.
    """
    is_partitioned = os.path.isdir(os.path.join(data_root, "partition_1"))
    if is_partitioned:
        mp_dir = os.path.join(data_root, "partition_1", "message_pool")
    else:
        mp_dir = os.path.join(data_root, "message_pool")
    hg_dir = os.path.join(mp_dir, "multi_scale_hypergraph")
    return mp_dir, hg_dir, is_partitioned


# ============================================================================
# Retrieval quality metrics
# ============================================================================

def _compute_recall_top1(
    cands: Dict[int, List[int]], ref: Dict[int, int]
) -> Tuple[float, float]:
    recall = top1 = 0
    for kg1, gt in ref.items():
        if kg1 not in cands:
            continue
        pool = cands[kg1]
        if gt in pool:
            recall += 1
        if pool and pool[0] == gt:
            top1 += 1
    denom = len(ref)
    return recall / denom, top1 / denom


def _compute_multi_scale_top1(
    hg_dir: str, ref: Dict[int, int], ablation: Optional[str] = None
) -> Tuple[float, float, int]:
    l1 = load_candidates(os.path.join(hg_dir, "L1_hypergraph.txt"))
    l2 = load_candidates(os.path.join(hg_dir, "L2_hypergraph.txt"))
    l3 = load_candidates(os.path.join(hg_dir, "L3_hypergraph.txt"))

    if ablation == "single_scale":
        l2 = {k: list(v) for k, v in l1.items()}
        l3 = {k: list(v) for k, v in l1.items()}

    n_cov = multi_hits = l1_hits = 0
    for kg1, gt in ref.items():
        if not ((kg1 in l1) or (kg1 in l2) or (kg1 in l3)):
            continue
        n_cov += 1
        t1, t2, t3 = l1.get(kg1, [None])[0], l2.get(kg1, [None])[0], l3.get(kg1, [None])[0]
        if t1 == gt:
            l1_hits += 1
        if any(x == gt for x in [t1, t2, t3]):
            multi_hits += 1

    denom = n_cov if n_cov > 0 else 1
    return multi_hits / denom, l1_hits / denom, n_cov


def task_retrieval_quality(data_root: str):
    """
    Multi-scale hypergraph retrieval quality.
    Metrics: Recall@K, Top-1 Precision across L1/L2/L3 and their union,
             plus w/ Single-Scale (ℓ=1) ablation.
    """
    ref = load_ref_pairs(data_root)
    if not ref:
        print("[ERROR] No ref_pairs found.")
        return

    mp_dir, hg_dir, is_part = _resolve_dirs(data_root)
    label = "partition_1" if is_part else os.path.basename(data_root.rstrip('/'))

    print("=" * 68)
    print("Multi-Scale Hypergraph Retrieval Quality")
    print(f"  Data root : {data_root}")
    print(f"  Layout    : {'partitioned' if is_part else 'flat'}")
    print(f"  Ref pairs : {len(ref)}")
    print("=" * 68)

    l1_file = os.path.join(hg_dir, "L1_hypergraph.txt")
    if not os.path.exists(l1_file):
        print("[SKIP] L1_hypergraph.txt not found.")
        return

    r1, p1 = _compute_recall_top1(load_candidates(os.path.join(hg_dir, "L1_hypergraph.txt")), ref)
    r2, p2 = _compute_recall_top1(load_candidates(os.path.join(hg_dir, "L2_hypergraph.txt")), ref)
    r3, p3 = _compute_recall_top1(load_candidates(os.path.join(hg_dir, "L3_hypergraph.txt")), ref)
    multi_top1, l1_only_top1, n_cov = _compute_multi_scale_top1(hg_dir, ref)
    multi_top1_ab, _, _ = _compute_multi_scale_top1(hg_dir, ref, ablation="single_scale")

    print(f"\n  [{label}] Hypergraph Results (n={n_cov} entities with coverage):")
    print(f"    L1 (relational) Recall@K : {r1*100:.1f}%  Top-1 Prec: {p1*100:.1f}%")
    print(f"    L2 (temporal)  Recall@K : {r2*100:.1f}%  Top-1 Prec: {p2*100:.1f}%")
    print(f"    L3 (unified)   Recall@K : {r3*100:.1f}%  Top-1 Prec: {p3*100:.1f}%")
    print(f"    MultiScale Top-1 Prec    : {multi_top1*100:.1f}%  (union of L1/L2/L3)")
    print(f"    w/ Single-Scale Hypergraph (ℓ=1): {multi_top1_ab*100:.1f}%  (ablation)")

    print(f"\n  Summary:")
    print(f"    {'Method':<45} {'Recall':>8} {'Top-1 Prec':>10}")
    print(f"    {'-'*45} {'-'*8} {'-'*10}")
    print(f"    {'HyDRA (full multi-scale Gm)':<45} {r1*100:>7.1f}% {multi_top1*100:>9.1f}%")
    print(f"    {'  - w/ Single-Scale Hypergraph (ℓ=1)':<45} {r1*100:>7.1f}% {multi_top1_ab*100:>9.1f}%")
    print(f"\n    Multi-scale gain: +{(multi_top1 - multi_top1_ab)*100:.1f}pp")
    print("=" * 68)


def task_pseudo_precision(data_root: str):
    """
    Pseudo-label precision.
    Precision = |fusion_pairs ∩ ref_pairs| / |fusion_pairs|
    Reads multi_scale_fusion_results.txt + ref_pairs.
    """
    ref = load_ref_pairs(data_root)
    if not ref:
        print("[ERROR] No ref_pairs found.")
        return

    _, _, is_part = _resolve_dirs(data_root)
    if is_part:
        fusion_file = os.path.join(data_root, "partition_1", "message_pool",
                                  "multi_scale_fusion_results.txt")
    else:
        fusion_file = os.path.join(data_root, "message_pool",
                                  "multi_scale_fusion_results.txt")

    if not os.path.exists(fusion_file):
        print(f"[SKIP] {fusion_file} not found.")
        return

    gt = set((int(k), int(v)) for k, v in ref.items())
    pred = set()
    with open(fusion_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    pred.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    continue

    correct = len(pred & gt)
    total = len(pred)
    precision = correct / total if total > 0 else 0.0

    print("=" * 68)
    print("Pseudo-label Precision")
    print(f"  Fusion file : {fusion_file}")
    print("=" * 68)
    print(f"\n  Predicted pairs   : {total}")
    print(f"  Correct (∩ ref)   : {correct}")
    print(f"  Precision         : {precision*100:.2f}%")
    print("=" * 68)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation: retrieval quality and pseudo-label precision. "
                    "No pipeline changes required."
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="e.g., /path/to/data/icews_wiki  or  /path/to/data/1m/en_de"
    )
    parser.add_argument(
        "--task", type=str,
        choices=["retrieval_quality", "pseudo_precision", "all"],
        default="all",
        help="Which metric to evaluate (default: all)"
    )
    args = parser.parse_args()

    print()

    if args.task in ("retrieval_quality", "all"):
        task_retrieval_quality(args.data_root)
        print()

    if args.task in ("pseudo_precision", "all"):
        task_pseudo_precision(args.data_root)
        print()


if __name__ == "__main__":
    main()
