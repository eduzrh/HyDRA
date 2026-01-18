#!/usr/bin/env python3
"""
Simplified multi-scale fusion (for ablation experiments)
Does not use advanced features such as interaction augmentation and conflict detection
"""

import os
from collections import defaultdict
from typing import List, Tuple, Dict


def multi_scale_fusion_simplified(data_dir, output_file=None, ablation_config=None):
    """
    Simplified multi-scale fusion (ablation experiment version).
    
    Args:
        data_dir: Data directory path
        output_file: Output file path
        ablation_config: Ablation experiment configuration
    
    Returns:
        List[Tuple[int, int]]: List of aligned entity pairs
    """
    print("⚠️  [ABLATION] Using simplified fusion (without interaction and conflict detection)")
    
    # Directly use L1 scale top-1 candidates as alignment results (simplest fusion method)
    message_pool_dir = os.path.join(data_dir, "message_pool")
    multi_scale_dir = os.path.join(message_pool_dir, "multi_scale_hypergraph")
    
    # Load L1 scale entity pairs
    l1_file = os.path.join(multi_scale_dir, "L1_hypergraph.txt")
    if not os.path.exists(l1_file):
        # If no L1 file, try loading from integration_top_pair.txt
        l1_file = os.path.join(message_pool_dir, "integration_top_pair.txt")
    
    if not os.path.exists(l1_file):
        print(f"Error: L1 scale file not found: {l1_file}")
        return []
    
    aligned_pairs = []
    
    # For each KG1 entity, only take the first candidate from L1 scale
    kg1_to_candidates = defaultdict(list)
    with open(l1_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    kg1_id = int(parts[0])
                    kg2_id = int(parts[1])
                    kg1_to_candidates[kg1_id].append(kg2_id)
                except ValueError:
                    continue
    
    # Only take the first candidate for each KG1 entity
    for kg1_id, candidates in kg1_to_candidates.items():
        if candidates:
            aligned_pairs.append((kg1_id, candidates[0]))
    
    # Save results
    if output_file is None:
        output_file = os.path.join(message_pool_dir, "multi_scale_fusion_results.txt")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in aligned_pairs:
            f.write(f"{kg1_id}\t{kg2_id}\n")
    
    print(f"Simplified fusion completed: {len(aligned_pairs)} aligned pairs")
    
    # Add to sup_pairs
    from multi_scale_fusion.multi_scale_fusion import add_to_sup_pairs
    if aligned_pairs:
        added_count = add_to_sup_pairs(data_dir, aligned_pairs)
        if added_count > 0:
            print(f"✓ Successfully added {added_count} new pairs to sup_pairs")
    
    return aligned_pairs

