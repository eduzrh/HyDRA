#!/usr/bin/env python3
"""
Relation Alignment Script

Features:
1. Load relation names from KG1 and KG2
2. Find similar/equivalent relation pairs through multiple methods:
   - Text similarity (based on relation names)
   - Co-occurrence patterns (relation co-occurrence in entity pairs)
   - Structural similarity (head-tail entity distribution of relations)

Input:
- data_dir: Data directory path
- rel_ids_1: KG1 relation file
- rel_ids_2: KG2 relation file
- entity_pairs: Entity alignment pair file (optional, for co-occurrence analysis)

Output:
- relation_alignment.txt: Relation alignment results
"""

import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher
import argparse

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


def load_relations(rel_file_path):
    """
    Load relation ID to relation name mapping
    
    Args:
        rel_file_path: Relation file path
        
    Returns:
        dict: {rel_id: rel_name}
    """
    relations = {}
    if not os.path.exists(rel_file_path):
        print(f"Warning: Relation file not found: {rel_file_path}")
        return relations
    
    with open(rel_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                rel_id = int(parts[0])
                rel_name = parts[1].strip()
                relations[rel_id] = rel_name
    
    print(f"Loaded {len(relations)} relations from {rel_file_path}")
    return relations


def load_triples(triples_file_path):
    """
    Load triples
    
    Args:
        triples_file_path: Triples file path
        
    Returns:
        list: [(head, rel, tail, time_start, time_end), ...]
    """
    triples = []
    if not os.path.exists(triples_file_path):
        print(f"Warning: Triples file not found: {triples_file_path}")
        return triples
    
    with open(triples_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                head = int(parts[0])
                rel = int(parts[1])
                tail = int(parts[2])
                time_start = int(parts[3]) if len(parts) > 3 else 0
                time_end = int(parts[4]) if len(parts) > 4 else time_start
                triples.append((head, rel, tail, time_start, time_end))
    
    print(f"Loaded {len(triples)} triples from {triples_file_path}")
    return triples


def load_entity_pairs(pairs_file_path):
    """
    Load entity alignment pairs
    
    Args:
        pairs_file_path: Entity pairs file path
        
    Returns:
        list: [(kg1_id, kg2_id), ...]
    """
    pairs = []
    if not os.path.exists(pairs_file_path):
        print(f"Warning: Entity pairs file not found: {pairs_file_path}")
        return pairs
    
    with open(pairs_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                kg1_id = int(parts[0])
                kg2_id = int(parts[1])
                pairs.append((kg1_id, kg2_id))
    
    print(f"Loaded {len(pairs)} entity pairs from {pairs_file_path}")
    return pairs


def text_similarity(name1, name2):
    """
    Calculate text similarity between two relation names
    
    Args:
        name1: Relation name 1
        name2: Relation name 2
        
    Returns:
        float: Similarity score (0-1)
    """
    # Convert to lowercase for comparison
    name1_lower = name1.lower()
    name2_lower = name2.lower()
    
    # If exactly the same
    if name1_lower == name2_lower:
        return 1.0
    
    # Calculate similarity using SequenceMatcher
    similarity = SequenceMatcher(None, name1_lower, name2_lower).ratio()
    
    # Check keyword matching
    words1 = set(name1_lower.split())
    words2 = set(name2_lower.split())
    
    # Remove common stop words
    stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'or', 'and'}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if words1 and words2:
        # Jaccard similarity
        word_overlap = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
        # Combined similarity (weighted average)
        similarity = 0.6 * similarity + 0.4 * word_overlap
    
    return similarity


def compute_cooccurrence_patterns(kg1_triples, kg2_triples, entity_pairs):
    """
    Calculate co-occurrence patterns of relations in entity pairs
    
    Args:
        kg1_triples: KG1 triples list
        kg2_triples: KG2 triples list
        entity_pairs: Entity alignment pairs list
        
    Returns:
        dict: {(kg1_rel_id, kg2_rel_id): cooccurrence_count}
    """
    # Build entity pair mapping
    kg1_to_kg2 = {kg1_id: kg2_id for kg1_id, kg2_id in entity_pairs}
    kg2_to_kg1 = {kg2_id: kg1_id for kg1_id, kg2_id in entity_pairs}
    
    # Build KG1 entity-relation mapping {kg1_entity_id: {rel_id: set(tail_ids)}}
    kg1_entity_rels = defaultdict(lambda: defaultdict(set))
    for head, rel, tail, _, _ in kg1_triples:
        kg1_entity_rels[head][rel].add(tail)
    
    # Build KG2 entity-relation mapping {kg2_entity_id: {rel_id: set(tail_ids)}}
    kg2_entity_rels = defaultdict(lambda: defaultdict(set))
    for head, rel, tail, _, _ in kg2_triples:
        kg2_entity_rels[head][rel].add(tail)
    
    # Calculate co-occurrence
    cooccurrence = defaultdict(int)
    
    for kg1_id, kg2_id in entity_pairs:
        kg1_rels = kg1_entity_rels.get(kg1_id, {})
        kg2_rels = kg2_entity_rels.get(kg2_id, {})
        
        # For each KG1 relation, find possible corresponding relations in KG2
        for kg1_rel, kg1_tails in kg1_rels.items():
            for kg2_rel, kg2_tails in kg2_rels.items():
                # If two relations have the same tail entity (through entity alignment mapping), they co-occur
                kg1_tail_set = set(kg1_tails)
                kg2_tail_set = set(kg2_tails)
                
                # Check if there are aligned tail entities
                aligned_tails = False
                for kg1_tail in kg1_tail_set:
                    if kg1_tail in kg1_to_kg2:
                        kg2_aligned_tail = kg1_to_kg2[kg1_tail]
                        if kg2_aligned_tail in kg2_tail_set:
                            aligned_tails = True
                            break
                
                if aligned_tails:
                    cooccurrence[(kg1_rel, kg2_rel)] += 1
    
    return cooccurrence


def find_relation_alignments(data_dir, use_cooccurrence=True, text_sim_threshold=0.3):
    """
    Find relation alignment pairs
    
    Args:
        data_dir: Data directory path
        use_cooccurrence: Whether to use co-occurrence patterns
        text_sim_threshold: Text similarity threshold
        
    Returns:
        list: [(kg1_rel_id, kg2_rel_id, score, method), ...]
    """
    print("\n" + "=" * 80)
    print("Relation Alignment: Finding Similar Relations")
    print("=" * 80 + "\n")
    
    # Load relations
    rel_ids_1_path = os.path.join(data_dir, "rel_ids_1")
    rel_ids_2_path = os.path.join(data_dir, "rel_ids_2")
    
    kg1_relations = load_relations(rel_ids_1_path)
    kg2_relations = load_relations(rel_ids_2_path)
    
    if not kg1_relations or not kg2_relations:
        print("Error: Failed to load relations")
        return []
    
    alignments = []
    
    # Method 1: Text similarity
    print("Method 1: Text Similarity Analysis...")
    text_alignments = []
    for kg1_rel_id, kg1_rel_name in kg1_relations.items():
        for kg2_rel_id, kg2_rel_name in kg2_relations.items():
            similarity = text_similarity(kg1_rel_name, kg2_rel_name)
            if similarity >= text_sim_threshold:
                text_alignments.append((kg1_rel_id, kg2_rel_id, similarity, "text_similarity"))
    
    # Sort by similarity
    text_alignments.sort(key=lambda x: x[2], reverse=True)
    print(f"  Found {len(text_alignments)} text-based alignments (threshold={text_sim_threshold})")
    alignments.extend(text_alignments)
    
    # Method 2: Co-occurrence patterns (if entity alignment pairs are provided)
    if use_cooccurrence:
        print("\nMethod 2: Co-occurrence Pattern Analysis...")
        pairs_file = os.path.join(data_dir, "message_pool", "integration_top_pair.txt")
        entity_pairs = load_entity_pairs(pairs_file)
        
        if entity_pairs:
            # Load triples
            triples_1_path = os.path.join(data_dir, "triples_1")
            triples_2_path = os.path.join(data_dir, "triples_2")
            
            kg1_triples = load_triples(triples_1_path)
            kg2_triples = load_triples(triples_2_path)
            
            if kg1_triples and kg2_triples:
                cooccurrence = compute_cooccurrence_patterns(kg1_triples, kg2_triples, entity_pairs)
                
                # Normalize co-occurrence scores (using logarithmic scaling)
                max_cooccur = max(cooccurrence.values()) if cooccurrence else 1
                for (kg1_rel_id, kg2_rel_id), count in cooccurrence.items():
                    # Normalize to 0-1 range
                    normalized_score = min(1.0, count / max_cooccur) if max_cooccur > 0 else 0.0
                    alignments.append((kg1_rel_id, kg2_rel_id, normalized_score, "cooccurrence"))
                
                print(f"  Found {len(cooccurrence)} co-occurrence alignments")
        else:
            print("  Skipped: No entity pairs found for co-occurrence analysis")
    
    # Deduplicate and merge identical relation pairs
    alignment_dict = {}
    for kg1_rel_id, kg2_rel_id, score, method in alignments:
        key = (kg1_rel_id, kg2_rel_id)
        if key not in alignment_dict:
            alignment_dict[key] = (kg1_rel_id, kg2_rel_id, score, method)
        else:
            # If already exists, take the higher score
            old_score = alignment_dict[key][2]
            if score > old_score:
                alignment_dict[key] = (kg1_rel_id, kg2_rel_id, score, method)
    
    # Sort by score
    final_alignments = sorted(alignment_dict.values(), key=lambda x: x[2], reverse=True)
    
    print(f"\nTotal unique alignments: {len(final_alignments)}")
    
    return final_alignments


def save_relation_alignments(alignments, kg1_relations, kg2_relations, output_file):
    """
    Save relation alignment results
    
    Args:
        alignments: Alignment results list
        kg1_relations: KG1 relations dictionary
        kg2_relations: KG2 relations dictionary
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KG1_Rel_ID\tKG1_Rel_Name\tKG2_Rel_ID\tKG2_Rel_Name\tScore\tMethod\n")
        for kg1_rel_id, kg2_rel_id, score, method in alignments:
            kg1_name = kg1_relations.get(kg1_rel_id, f"Unknown_{kg1_rel_id}")
            kg2_name = kg2_relations.get(kg2_rel_id, f"Unknown_{kg2_rel_id}")
            f.write(f"{kg1_rel_id}\t{kg1_name}\t{kg2_rel_id}\t{kg2_name}\t{score:.4f}\t{method}\n")
    
    print(f"\nSaved relation alignments to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Relation Alignment Tool")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: data_dir/message_pool/relation_alignment.txt)")
    parser.add_argument("--text_threshold", type=float, default=0.3, help="Text similarity threshold (default: 0.3)")
    parser.add_argument("--no_cooccurrence", action="store_true", help="Disable co-occurrence analysis")
    
    args = parser.parse_args()
    
    # Find relation alignments
    alignments = find_relation_alignments(
        args.data_dir,
        use_cooccurrence=not args.no_cooccurrence,
        text_sim_threshold=args.text_threshold
    )
    
    if not alignments:
        print("\nNo relation alignments found.")
        return
    
    # Load relation names for output
    rel_ids_1_path = os.path.join(args.data_dir, "rel_ids_1")
    rel_ids_2_path = os.path.join(args.data_dir, "rel_ids_2")
    kg1_relations = load_relations(rel_ids_1_path)
    kg2_relations = load_relations(rel_ids_2_path)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.data_dir, "message_pool", "relation_alignment.txt")
    
    save_relation_alignments(alignments, kg1_relations, kg2_relations, output_file)
    
    # Print top 10 alignment results
    print("\n" + "=" * 80)
    print("Top 10 Relation Alignments:")
    print("=" * 80)
    for i, (kg1_rel_id, kg2_rel_id, score, method) in enumerate(alignments[:10], 1):
        kg1_name = kg1_relations.get(kg1_rel_id, f"Unknown_{kg1_rel_id}")
        kg2_name = kg2_relations.get(kg2_rel_id, f"Unknown_{kg2_rel_id}")
        print(f"{i:2d}. KG1[{kg1_rel_id}] '{kg1_name}' <-> KG2[{kg2_rel_id}] '{kg2_name}' (score={score:.4f}, method={method})")


if __name__ == "__main__":
    main()

