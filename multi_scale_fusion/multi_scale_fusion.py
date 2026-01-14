#!/usr/bin/env python3
"""
Multi-Scale Interactive Enhancement Fusion Module

Features:
1. Read hypergraph entity pair files from L1, L2, L3 scales
2. For each KG1 entity, collect candidate entities from three scales
3. Use large language model to judge candidate entities and select best aligned entity
4. Output results from large language model judgment

Reference: Logic from LLM_executor.py
"""

import os
import sys
import queue
import threading
from collections import defaultdict
from tqdm import tqdm
import httpx
from openai import OpenAI

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import intra-scale interaction and conflict detection modules
try:
    from intra_scale_interaction import (
        analyze_intra_scale_interaction,
        detect_cross_scale_conflicts,
        detect_intra_scale_conflicts,
        generate_conflict_summary
    )
except ImportError:
    # If relative import fails, try importing from current directory
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from intra_scale_interaction import (
        analyze_intra_scale_interaction,
        detect_cross_scale_conflicts,
        detect_intra_scale_conflicts,
        generate_conflict_summary
    )

# Try importing ThreadPoolExecutor
try:
    from ThreadPoolExecutor import ThreadPoolExecutor
except ImportError:
    from concurrent.futures import ThreadPoolExecutor

# Try importing tokens_cal (if exists)
try:
    import tokens_cal
except ImportError:
    # If tokens_cal doesn't exist, create a simple placeholder
    class tokens_cal:
        @staticmethod
        def update_add_var(value):
            pass


def load_entity_names(file_path):
    """Load entity ID to name mapping"""
    entity_names = {}
    if not os.path.exists(file_path):
        print(f"Warning: Entity file not found: {file_path}")
        return entity_names
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    entity_id = int(parts[0])
                    entity_name = parts[1]
                    entity_names[entity_id] = entity_name
                except ValueError:
                    continue
    return entity_names


def load_multi_scale_pairs(data_dir):
    """
    Load entity pairs from multi-scale hypergraphs
    
    Args:
        data_dir: Data directory path
        
    Returns:
        dict: {kg1_entity_id: {'L1': [kg2_ids], 'L2': [kg2_ids], 'L3': [kg2_ids]}}
    """
    message_pool_dir = os.path.join(data_dir, "message_pool")
    multi_scale_dir = os.path.join(message_pool_dir, "multi_scale_hypergraph")
    
    scale_files = {
        'L1': os.path.join(multi_scale_dir, "L1_hypergraph.txt"),
        'L2': os.path.join(multi_scale_dir, "L2_hypergraph.txt"),
        'L3': os.path.join(multi_scale_dir, "L3_hypergraph.txt")
    }
    
    multi_scale_pairs = defaultdict(lambda: {'L1': [], 'L2': [], 'L3': []})
    
    for scale, file_path in scale_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: {scale} scale file not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        kg1_id = int(parts[0])
                        kg2_id = int(parts[1])
                        multi_scale_pairs[kg1_id][scale].append(kg2_id)
                    except ValueError:
                        continue
    
    print(f"Loaded multi-scale pairs for {len(multi_scale_pairs)} KG1 entities")
    for kg1_id, scales in list(multi_scale_pairs.items())[:5]:
        print(f"  Entity {kg1_id}: L1={len(scales['L1'])}, L2={len(scales['L2'])}, L3={len(scales['L3'])}")
    
    return multi_scale_pairs


def load_triples(file_path):
    """Load triples data"""
    triples = []
    if not os.path.exists(file_path):
        return triples
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    continue
    return triples


def load_relation_names(file_path):
    """Load relation names"""
    rel_names = {}
    if not os.path.exists(file_path):
        return rel_names
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    rel_id = int(parts[0])
                    rel_name = parts[1]
                    rel_names[rel_id] = rel_name
                except ValueError:
                    continue
    return rel_names


def get_entity_context(entity_id, entity_names, triples=None, rel_names=None, n=3):
    """
    Get entity context information (including relation information for semantic judgment)
    
    Args:
        entity_id: Entity ID
        entity_names: Entity name dictionary
        triples: Triples list (optional)
        rel_names: Relation name dictionary (optional)
        n: Return top n relations
        
    Returns:
        str: Entity context string
    """
    context = f"Entity Name: {entity_names.get(entity_id, f'Unknown (ID: {entity_id})')}"
    
    # Add relation information (if available)
    if triples and rel_names:
        relations = []
        for h, r, t in triples:
            if h == entity_id:
                rel_str = rel_names.get(r, f"relation_{r}")
                tail_str = entity_names.get(t, f"entity_{t}")
                relations.append(f"- Has relation '{rel_str}' with {tail_str}")
            elif t == entity_id:
                rel_str = rel_names.get(r, f"relation_{r}")
                head_str = entity_names.get(h, f"entity_{h}")
                relations.append(f"- Is '{rel_str}' of {head_str}")
            if len(relations) >= n:
                break
        
        if relations:
            context += "\nRelationships:\n" + "\n".join(relations[:n])
        else:
            context += "\nRelationships: No relationships found"
    
    return context


def multi_scale_fusion(data_dir, output_file=None):
    """
    Multi-scale interactive enhancement fusion
    
    Args:
        data_dir: Data directory path
        output_file: Output file path (default: message_pool/multi_scale_fusion_results.txt)
    """
    print("\n" + "=" * 80)
    print("Multi-Scale Fusion: Multi-Scale Interactive Enhancement")
    print("=" * 80 + "\n")
    
    # Set file paths
    message_pool_dir = os.path.join(data_dir, "message_pool")
    if output_file is None:
        output_file = os.path.join(message_pool_dir, "multi_scale_fusion_results.txt")
    
    # Load entity names
    ent_ids_1_path = os.path.join(data_dir, 'ent_ids_1')
    ent_ids_2_path = os.path.join(data_dir, 'ent_ids_2')
    ent_names_1 = load_entity_names(ent_ids_1_path)
    ent_names_2 = load_entity_names(ent_ids_2_path)
    
    # Load triples and relation names (for semantic judgment)
    triples_1_path = os.path.join(data_dir, 'triples_1')
    triples_2_path = os.path.join(data_dir, 'triples_2')
    rel_ids_1_path = os.path.join(data_dir, 'rel_ids_1')
    rel_ids_2_path = os.path.join(data_dir, 'rel_ids_2')
    
    triples_1 = load_triples(triples_1_path)
    triples_2 = load_triples(triples_2_path)
    rel_names_1 = load_relation_names(rel_ids_1_path)
    rel_names_2 = load_relation_names(rel_ids_2_path)
    
    print(f"Loaded {len(ent_names_1)} KG1 entities")
    print(f"Loaded {len(ent_names_2)} KG2 entities")
    print(f"Loaded {len(triples_1)} KG1 triples")
    print(f"Loaded {len(triples_2)} KG2 triples")
    print(f"Loaded {len(rel_names_1)} KG1 relations")
    print(f"Loaded {len(rel_names_2)} KG2 relations")
    
    # Load multi-scale entity pairs
    multi_scale_pairs = load_multi_scale_pairs(data_dir)
    
    if not multi_scale_pairs:
        print("Error: No multi-scale pairs found. Please run s4_to_retrieval.py first.")
        return []
    
    # Setup OpenAI client
    # Note: API credentials should be configured via environment variables or config file
    # Example: client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = None  # TODO: Configure OpenAI client with proper credentials
    
    # LLM Agent Profile
    LLM_Agent_Profile = '''
Goal: As a knowledge graph alignment expert, determine if the first entity represents the SAME REAL-WORLD OBJECT as one of the candidate entities from multiple scales (L1, L2, L3).

Core Task: Semantic Consistency Judgment
- You need to judge whether two entities refer to the SAME REAL-WORLD OBJECT based on their semantic meaning
- Consider entity names, descriptions, and any available context
- Focus on semantic equivalence, not just ID matching
- Two entities are aligned if they represent the same real-world entity, even if their names or IDs differ

Constraint: 
- If there is a semantically matching entity (same real-world object), return ONLY the ID of the matching candidate entity
- If none of them matches (not the same real-world object), return ONLY "No"
- Do NOT include any explanation, only return the entity ID or "No"
    '''
    
    aligned_pairs = []
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=30)
    result_queue = queue.Queue()
    
    def fusion_task(kg1_entity, candidate_info):
        """
        Perform multi-scale fusion judgment for a single KG1 entity
        
        Args:
            kg1_entity: KG1 entity ID
            candidate_info: {'L1': [kg2_ids], 'L2': [kg2_ids], 'L3': [kg2_ids]}
        """
        try:
            # Collect all candidate entities (deduplicated)
            all_candidates = set()
            scale_to_candidates = {}
            
            for scale in ['L1', 'L2', 'L3']:
                candidates = candidate_info.get(scale, [])
                scale_to_candidates[scale] = candidates
                all_candidates.update(candidates)
            
            if not all_candidates:
                return
            
            # ===== Intra-scale interaction and conflict detection =====
            # 1. Analyze intra-scale interaction
            scale_analysis = analyze_intra_scale_interaction(candidate_info, ent_names_2)
            
            # 2. Detect cross-scale conflicts
            cross_scale_conflicts = detect_cross_scale_conflicts(candidate_info, scale_analysis)
            
            # 3. Detect intra-scale conflicts
            intra_scale_conflicts = detect_intra_scale_conflicts(candidate_info, scale_analysis)
            
            # 4. Generate conflict summary
            conflict_summary = generate_conflict_summary(
                cross_scale_conflicts, 
                intra_scale_conflicts, 
                scale_analysis
            )
            # ===== End conflict detection =====
            
            # Build KG1 entity context (including relation information for semantic judgment)
            context1 = get_entity_context(kg1_entity, ent_names_1, triples_1, rel_names_1, n=5)
            
            # Build candidate entity contexts (grouped by scale, considering intra-scale ranking, including relation information)
            candidates_contexts = []
            for scale in ['L1', 'L2', 'L3']:
                candidates = scale_to_candidates.get(scale, [])
                scale_conf = scale_analysis.get(scale, {}).get('confidence_signal', 'unknown')
                for rank, kg2_entity in enumerate(candidates, 1):
                    # Get candidate entity context (including relation information)
                    context2 = get_entity_context(kg2_entity, ent_names_2, triples_2, rel_names_2, n=5)
                    # Add ranking information
                    rank_info = f" (Rank {rank} in {scale}, confidence: {scale_conf})" if rank == 1 else f" (Rank {rank} in {scale})"
                    candidates_contexts.append({
                        'entity_id': kg2_entity,
                        'scale': scale,
                        'rank': rank,
                        'context': context2 + rank_info
                    })
            
            # Build Prompt (including conflict information)
            prompt = LLM_Agent_Profile + f"""

Entity 1 (KG1, ID: {kg1_entity}):
{context1}

{conflict_summary}

Candidate entities from multiple scales:"""
            
            for i, candidate in enumerate(candidates_contexts, 1):
                prompt += f"\n\nCandidate {i} (KG2, ID: {candidate['entity_id']}, from {candidate['scale']} scale):\n{candidate['context']}"
            
            prompt += """\n\nSemantic Consistency Judgment:
Do any of these candidate entities represent the SAME REAL-WORLD OBJECT as Entity 1?

Please judge based on SEMANTIC MEANING:
1. Compare entity names - are they referring to the same real-world entity?
2. Compare relationships - do they have similar semantic relationships?
3. Consider the consistency across different scales (L1, L2, L3)
4. Pay attention to any conflicts detected above
5. Prefer candidates with higher consensus across scales

IMPORTANT: 
- Focus on SEMANTIC EQUIVALENCE (same real-world object), not just ID matching
- Two entities are aligned if they represent the same real-world entity semantically
- If there is a semantically matching entity, return ONLY the entity ID (e.g., "26471")
- If none of them match (not the same real-world object), return ONLY "No"
- Do NOT include any explanation, only return the entity ID or "No":"""
            
            # Call LLM
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Update token count
            try:
                tokens_cal.update_add_var(response.usage.total_tokens)
            except:
                pass
            
            print(f"Processing entity {kg1_entity} with {len(all_candidates)} candidates from {len([s for s in ['L1','L2','L3'] if scale_to_candidates.get(s)])} scales")
            print(f"Answer: {answer}")
            
            # Parse answer
            if answer.lower() != "no":
                # Try to extract entity ID
                for kg2_id in all_candidates:
                    if str(kg2_id) in answer:
                        with lock:
                            result_queue.put((kg1_entity, kg2_id))
                            aligned_pairs.append((kg1_entity, kg2_id))
                        break
        
        except Exception as e:
            print(f"Error processing entity {kg1_entity}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Process each KG1 entity
    print(f"\nStarting multi-scale fusion for {len(multi_scale_pairs)} entities...")
    for kg1_entity, candidate_info in tqdm(multi_scale_pairs.items(), desc="Submitting tasks"):
        executor.submit(fusion_task, kg1_entity, candidate_info)
    
    # Wait for all tasks to complete
    executor.shutdown(wait=True)
    
    # Write results to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as output_f:
        while not result_queue.empty():
            kg1_entity, kg2_id = result_queue.get()
            output_f.write(f"{kg1_entity}\t{kg2_id}\n")
            output_f.flush()
    
    # Deduplicate and get alignment results
    unique_pairs = deduplicate_output_file(output_file)
    aligned_pairs = list(unique_pairs) if unique_pairs else []
    
    print(f"\n" + "=" * 80)
    print(f"Multi-scale fusion completed!")
    print(f"Output file: {output_file}")
    print(f"Total aligned pairs: {len(aligned_pairs)}")
    print("=" * 80 + "\n")
    
    # Add fusion results to sup_pairs
    if aligned_pairs:
        print("Adding fusion results to sup_pairs...")
        added_count = add_to_sup_pairs(data_dir, aligned_pairs)
        if added_count > 0:
            print(f"✓ Successfully added {added_count} new pairs to sup_pairs")
        print()
    
    return aligned_pairs


def deduplicate_output_file(file_path):
    """Deduplicate output file"""
    if not os.path.exists(file_path):
        return set()
    
    # Read all lines and deduplicate
    unique_pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    e1 = int(parts[0])
                    e2 = int(parts[1])
                    unique_pairs.add((e1, e2))
                except ValueError:
                    continue
    
    # Rewrite deduplicated results
    with open(file_path, 'w', encoding='utf-8') as f:
        for e1, e2 in sorted(unique_pairs):
            f.write(f"{e1}\t{e2}\n")
    
    print(f"Deduplicated file {file_path}: {len(unique_pairs)} unique pairs")
    return unique_pairs


def add_to_sup_pairs(data_dir, aligned_pairs):
    """
    Selectively add fusion results to sup_pairs
    
    Rule: A pair (KG1 entity and KG2 entity) can be added if neither has appeared in sup_pairs
    - KG1 entity has not appeared in sup_pairs (as first column)
    - KG2 entity has not appeared in sup_pairs (as second column)
    
    Args:
        data_dir: Data directory path
        aligned_pairs: List of aligned entity pairs [(kg1_id, kg2_id), ...]
    
    Returns:
        int: Number of successfully added entity pairs
    """
    if not aligned_pairs:
        return 0
    
    sup_pairs_file = os.path.join(data_dir, 'sup_pairs')
    
    # Load existing sup_pairs
    sup_pairs = set()
    kg1_ids = set()
    kg2_ids = set()
    
    if os.path.exists(sup_pairs_file):
        with open(sup_pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        kg1_id = int(parts[0])
                        kg2_id = int(parts[1])
                        sup_pairs.add((kg1_id, kg2_id))
                        kg1_ids.add(kg1_id)
                        kg2_ids.add(kg2_id)
                    except ValueError:
                        continue
    
    # Filter eligible entity pairs
    new_pairs = []
    for kg1_id, kg2_id in aligned_pairs:
        # Check if KG1 entity already exists
        if kg1_id in kg1_ids:
            continue
        # Check if KG2 entity already exists
        if kg2_id in kg2_ids:
            continue
        # Check if completely duplicate
        if (kg1_id, kg2_id) in sup_pairs:
            continue
        # Eligible entity pairs
        new_pairs.append((kg1_id, kg2_id))
    
    if not new_pairs:
        print(f"  No new pairs to add to sup_pairs (all pairs already exist)")
        return 0
    
    # Merge all entity pairs
    all_pairs = sup_pairs | set(new_pairs)
    
    # Sort by KG1 entity ID, then by KG2 entity ID
    sorted_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))
    
    # Backup original file
    backup_file = sup_pairs_file + ".backup"
    if os.path.exists(sup_pairs_file):
        import shutil
        shutil.copy2(sup_pairs_file, backup_file)
        print(f"  Backup created: {backup_file}")
    
    # Write new file
    with open(sup_pairs_file, 'w', encoding='utf-8') as f:
        for kg1_id, kg2_id in sorted_pairs:
            f.write(f"{kg1_id}\t{kg2_id}\n")
    
    print(f"  Added {len(new_pairs)} new pairs to sup_pairs")
    print(f"  Updated sup_pairs: {len(sorted_pairs)} total pairs (original: {len(sup_pairs)}, added: {len(new_pairs)})")
    
    return len(new_pairs)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Scale Fusion: Multi-Scale Interactive Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 基本用法
  python multi_scale_fusion.py --data_dir /path/to/data/icews_wiki
  
  # 指定输出文件
  python multi_scale_fusion.py --data_dir /path/to/data/icews_wiki --output custom_output.txt
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory path (e.g., /path/to/data/icews_wiki)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, default: message_pool/multi_scale_fusion_results.txt)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Execute multi-scale fusion
    aligned_pairs = multi_scale_fusion(args.data_dir, args.output)
    
    if aligned_pairs:
        print(f"✓ Multi-scale fusion completed successfully!")
        print(f"  Found {len(aligned_pairs)} aligned entity pairs")
        sys.exit(0)
    else:
        print(f"✗ Multi-scale fusion completed with no aligned pairs")
        sys.exit(1)


if __name__ == "__main__":
    main()

