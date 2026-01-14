#!/usr/bin/env python3
"""
Standalone Relation Alignment Stage

Features:
1. Run relation alignment and generate relation alignment file
2. As an independent processing stage, can be run before hypergraph decomposition

Usage:
    python run_relation_alignment.py --data_dir data/icews_wiki
"""

import os
import sys
import argparse

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from relation_alignment import find_relation_alignments, load_relations, save_relation_alignments


def run_relation_alignment_stage(data_dir, text_threshold=0.4, use_cooccurrence=True):
    """
    Run relation alignment stage
    
    Args:
        data_dir: Data directory path
        text_threshold: Text similarity threshold
        use_cooccurrence: Whether to use co-occurrence patterns
        
    Returns:
        str: Relation alignment file path, returns None if failed
    """
    print("\n" + "=" * 80)
    print("Relation Alignment Stage")
    print("=" * 80 + "\n")
    
    # Check necessary files
    rel_ids_1_path = os.path.join(data_dir, "rel_ids_1")
    rel_ids_2_path = os.path.join(data_dir, "rel_ids_2")
    
    if not os.path.exists(rel_ids_1_path):
        print(f"Error: KG1 relation file not found: {rel_ids_1_path}")
        return None
    
    if not os.path.exists(rel_ids_2_path):
        print(f"Error: KG2 relation file not found: {rel_ids_2_path}")
        return None
    
    # Create output directory
    output_dir = os.path.join(data_dir, "message_pool")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "relation_alignment.txt")
    
    # Run relation alignment
    alignments = find_relation_alignments(
        data_dir,
        use_cooccurrence=use_cooccurrence,
        text_sim_threshold=text_threshold
    )
    
    if not alignments:
        print("\nWarning: No relation alignments found.")
        print("The relation alignment file will be empty or not created.")
        return output_file
    
    # Load relation names
    kg1_relations = load_relations(rel_ids_1_path)
    kg2_relations = load_relations(rel_ids_2_path)
    
    # Save results
    save_relation_alignments(alignments, kg1_relations, kg2_relations, output_file)
    
    # Statistics
    print("\n" + "=" * 80)
    print("Relation Alignment Statistics")
    print("=" * 80)
    
    # Statistics for one-to-many/many-to-one
    kg1_to_kg2 = {}
    kg2_to_kg1 = {}
    for kg1_rel_id, kg2_rel_id, score, method in alignments:
        if kg1_rel_id not in kg1_to_kg2:
            kg1_to_kg2[kg1_rel_id] = []
        kg1_to_kg2[kg1_rel_id].append((kg2_rel_id, score))
        
        if kg2_rel_id not in kg2_to_kg1:
            kg2_to_kg1[kg2_rel_id] = []
        kg2_to_kg1[kg2_rel_id].append((kg1_rel_id, score))
    
    one_to_many = {k: v for k, v in kg1_to_kg2.items() if len(v) > 1}
    many_to_one = {k: v for k, v in kg2_to_kg1.items() if len(v) > 1}
    
    print(f"Total alignments: {len(alignments)}")
    print(f"Unique KG1 relations with alignments: {len(kg1_to_kg2)}")
    print(f"Unique KG2 relations with alignments: {len(kg2_to_kg1)}")
    print(f"One-to-many alignments (KG1 -> multiple KG2): {len(one_to_many)}")
    print(f"Many-to-one alignments (multiple KG1 -> KG2): {len(many_to_one)}")
    
    if one_to_many:
        print(f"\nTop 3 one-to-many examples (KG1 -> multiple KG2):")
        for i, (kg1_rel_id, kg2_list) in enumerate(sorted(one_to_many.items(), key=lambda x: len(x[1]), reverse=True)[:3], 1):
            kg1_name = kg1_relations.get(kg1_rel_id, f"Unknown_{kg1_rel_id}")
            print(f"  {i}. KG1[{kg1_rel_id}] '{kg1_name}' -> {len(kg2_list)} KG2 relations")
    
    if many_to_one:
        print(f"\nTop 3 many-to-one examples (multiple KG1 -> KG2):")
        for i, (kg2_rel_id, kg1_list) in enumerate(sorted(many_to_one.items(), key=lambda x: len(x[1]), reverse=True)[:3], 1):
            kg2_name = kg2_relations.get(kg2_rel_id, f"Unknown_{kg2_rel_id}")
            print(f"  {i}. {len(kg1_list)} KG1 relations -> KG2[{kg2_rel_id}] '{kg2_name}'")
    
    print("\n" + "=" * 80)
    print(f"Relation alignment file saved: {output_file}")
    print("=" * 80 + "\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Run Relation Alignment Stage")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--text_threshold", type=float, default=0.4, help="Text similarity threshold (default: 0.4)")
    parser.add_argument("--no_cooccurrence", action="store_true", help="Disable co-occurrence analysis")
    
    args = parser.parse_args()
    
    output_file = run_relation_alignment_stage(
        args.data_dir,
        text_threshold=args.text_threshold,
        use_cooccurrence=not args.no_cooccurrence
    )
    
    if output_file:
        print("✓ Relation alignment stage completed successfully")
        return 0
    else:
        print("✗ Relation alignment stage failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


