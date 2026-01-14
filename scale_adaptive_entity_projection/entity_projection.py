#!/usr/bin/env python3
"""
S4 to Retrieval Connection Script

Features:
1. Read top-k results from Simple-HHEA output (integration_top_pair.txt)
2. Extract KG2 entity IDs and convert to retrieval document format (inrag_ent_ids_2_pre_embeding.txt)
3. Use KG1 entities as queries and call neural_retrieval for retrieval

Input:
- Top-k file from Simple-HHEA output: data/{dataset}/message_pool/integration_top_pair.txt
  Format: kg1_entity_id\tkg2_entity_id

Output:
- Retrieval document file: data/{dataset}/inrag_ent_ids_2_pre_embeding.txt
  Format: entity_id\tentity_name
- Retrieval results: data/{dataset}/message_pool/retriever_outputs.txt
  Format: kg1_entity_id\tkg2_entity_id
"""

import os
import sys
from collections import defaultdict

# Add project path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from multi_scale_hypergraph_retrieval.neural_retrieval import neural_retrieval
from multi_scale_hypergraph_retrieval.hypergraph_decomposition import run_hypergraph_decomposition
from scale_adaptive_entity_projection.run_relation_alignment import run_relation_alignment_stage


def load_entity_ids(entity_file_path):
    """
    Load entity ID to entity name mapping
    
    Args:
        entity_file_path: Entity file path (ent_ids_1 or ent_ids_2)
        
    Returns:
        dict: {entity_id: entity_name}
    """
    entity_dict = {}
    if not os.path.exists(entity_file_path):
        print(f"Warning: Entity file not found: {entity_file_path}")
        return entity_dict
    
    with open(entity_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                entity_id = parts[0].strip()
                entity_name = parts[1].strip()
                entity_dict[entity_id] = entity_name
    
    print(f"Loaded {len(entity_dict)} entities from {entity_file_path}")
    return entity_dict


def extract_entities_from_topk(topk_file_path):
    """
    Extract KG1 and KG2 entity ID sets from Simple-HHEA top-k output file
    
    Args:
        topk_file_path: Top-k result file path
        
    Returns:
        tuple: (kg1_entity_ids set, kg2_entity_ids set)
    """
    kg1_entity_ids = set()
    kg2_entity_ids = set()
    
    if not os.path.exists(topk_file_path):
        print(f"Warning: Top-k file not found: {topk_file_path}")
        return kg1_entity_ids, kg2_entity_ids
    
    with open(topk_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                kg1_id = parts[0].strip()
                kg2_id = parts[1].strip()
                kg1_entity_ids.add(kg1_id)
                kg2_entity_ids.add(kg2_id)
    
    print(f"Extracted {len(kg1_entity_ids)} unique KG1 entities from top-k file")
    print(f"Extracted {len(kg2_entity_ids)} unique KG2 entities from top-k file")
    return kg1_entity_ids, kg2_entity_ids


def extract_kg2_entities_from_topk(topk_file_path):
    """
    Extract KG2 entity ID set from Simple-HHEA top-k output file (for backward compatibility)
    
    Args:
        topk_file_path: Top-k result file path
        
    Returns:
        set: KG2 entity ID set
    """
    _, kg2_entity_ids = extract_entities_from_topk(topk_file_path)
    return kg2_entity_ids


def create_retrieval_document(data_dir, kg2_entity_ids, ent_ids_2_path, output_file_path):
    """
    Create retrieval document file (inrag_ent_ids_2_pre_embeding.txt)
    
    Args:
        data_dir: Data directory
        kg2_entity_ids: KG2 entity ID set
        ent_ids_2_path: KG2 entity file path
        output_file_path: Output file path
        
    Returns:
        int: Number of successfully written entities
    """
    # Load all KG2 entities
    all_kg2_entities = load_entity_ids(ent_ids_2_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file_path) if os.path.dirname(output_file_path) else '.', exist_ok=True)
    
    # Write retrieval document
    written_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entity_id in kg2_entity_ids:
            if entity_id in all_kg2_entities:
                entity_name = all_kg2_entities[entity_id]
                f.write(f"{entity_id}\t{entity_name}\n")
                written_count += 1
            else:
                print(f"Warning: Entity ID {entity_id} not found in ent_ids_2")
    
    print(f"Created retrieval document with {written_count} entities: {output_file_path}")
    return written_count


def create_retrieval_document_with_aspects(data_dir, kg2_entity_ids, aspect_ids, aspect_names, ent_ids_2_path, output_file_path):
    """
    Create retrieval document file (including original KG2 entities and aspect entities)
    
    Args:
        data_dir: Data directory
        kg2_entity_ids: Original KG2 entity ID set
        aspect_ids: Aspect entity ID set
        aspect_names: Aspect entity name dictionary {aspect_id: aspect_name}
        ent_ids_2_path: KG2 entity file path
        output_file_path: Output file path
        
    Returns:
        int: Number of successfully written entities
    """
    # Load all KG2 entities
    all_kg2_entities = load_entity_ids(ent_ids_2_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file_path) if os.path.dirname(output_file_path) else '.', exist_ok=True)
    
    written_count = 0
    
    # Write original KG2 entities and aspect entities
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 1. Write original KG2 entities
        for entity_id in kg2_entity_ids:
            if entity_id in all_kg2_entities:
                entity_name = all_kg2_entities[entity_id]
                f.write(f"{entity_id}\t{entity_name}\n")
                written_count += 1
        
        # 2. Write aspect entities
        for aspect_id in aspect_ids:
            aspect_name = aspect_names.get(aspect_id, f"Aspect_{aspect_id}")
            f.write(f"{aspect_id}\t{aspect_name}\n")
            written_count += 1
    
    print(f"Created retrieval document with {written_count} entities (original: {len(kg2_entity_ids)}, aspects: {len(aspect_ids)}): {output_file_path}")
    return written_count


def create_query_entities_file(data_dir, kg1_entity_ids, ent_ids_1_path, output_file_path):
    """
    Create query entities file (only contains KG1 entities from top-k results)
    
    Args:
        data_dir: Data directory
        kg1_entity_ids: KG1 entity ID set
        ent_ids_1_path: Original KG1 entity file path
        output_file_path: Output file path (temporary file)
        
    Returns:
        int: Number of successfully written entities
    """
    # Load all KG1 entities
    all_kg1_entities = load_entity_ids(ent_ids_1_path)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file_path) if os.path.dirname(output_file_path) else '.', exist_ok=True)
    
    # Write query entities file
    written_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entity_id in kg1_entity_ids:
            if entity_id in all_kg1_entities:
                entity_name = all_kg1_entities[entity_id]
                f.write(f"{entity_id}\t{entity_name}\n")
                written_count += 1
            else:
                print(f"Warning: Entity ID {entity_id} not found in ent_ids_1")
    
    print(f"Created query entities file with {written_count} entities: {output_file_path}")
    return written_count


def s4_to_retrieval(data_dir, dataset_name=None, iteration=1, force_update=False):
    """
    Convert Simple-HHEA top-k output to retrieval format and execute retrieval
    
    Process:
    1. Read Simple-HHEA top-k output (integration_top_pair.txt)
    2. Extract KG2 entity IDs and create retrieval document (inrag_ent_ids_2_pre_embeding.txt)
    3. Use KG1 entities as queries and call neural_retrieval for retrieval
    
    Args:
        data_dir: Data directory path (e.g., /path/to/data/icews_wiki)
        dataset_name: Dataset name (optional, extracted from data_dir path if not provided)
        iteration: Current iteration number (default: 1)
        force_update: Whether to force update projection coverage file and FAISS index (default: False, automatically set to True for iterations after the first)
    """
    print("\n" + "=" * 80)
    print("S4 to Retrieval: Connect Simple-HHEA output and Neural Retrieval")
    print("=" * 80 + "\n")
    
    # Get dataset name
    if dataset_name is None:
        dataset_name = os.path.basename(data_dir.rstrip('/'))
    
    # File paths
    topk_file_path = os.path.join(data_dir, "message_pool", "integration_top_pair.txt")
    ent_ids_1_path = os.path.join(data_dir, "ent_ids_1")
    ent_ids_2_path = os.path.join(data_dir, "ent_ids_2")
    retrieval_doc_path = os.path.join(data_dir, "inrag_ent_ids_2_pre_embeding.txt")
    
    # Step 1: Check if top-k file exists
    if not os.path.exists(topk_file_path):
        print(f"Error: Top-k file not found: {topk_file_path}")
        print("Please run Simple-HHEA first to generate the top-k results.")
        return False
    
    print(f"Step 1: Reading top-k results from: {topk_file_path}")
    
    # Step 1.5: Relation alignment (run before hypergraph decomposition)
    print(f"\nStep 1.5: Relation Alignment (before hypergraph decomposition)...")
    alignment_file = run_relation_alignment_stage(
        data_dir,
        text_threshold=0.4,
        use_cooccurrence=True
    )
    if alignment_file:
        print(f"  Relation alignment completed: {alignment_file}")
    else:
        print(f"  Warning: Relation alignment failed or returned empty results")
        print(f"  Hypergraph decomposition will proceed without relation alignment")
    
    # Step 2: Hypergraph decomposition (decompose entity pairs into multiple aspects of KG2 entities)
    print(f"\nStep 2: Hypergraph Decomposition (creating aspect entities)...")
    try:
        decomposition_result = run_hypergraph_decomposition(data_dir, os.path.join(data_dir, "message_pool"))
        
        if not decomposition_result:
            print("Error: Hypergraph decomposition failed.")
            return False
        
        # Get decomposed aspect entities
        aspect_ids = decomposition_result['aspect_ids']
        aspect_names = decomposition_result['aspect_names']
        aspect_mapping = decomposition_result['aspect_mapping']
        
        print(f"  Generated {len(aspect_ids)} aspect entities:")
        print(f"    - Temporal aspects: {len(decomposition_result['temporal_aspects'])}")
        print(f"    - Relational aspects: {len(decomposition_result['relational_aspects'])}")
    except Exception as e:
        print(f"Warning: Hypergraph decomposition failed: {str(e)}")
        print("  Continuing without aspect entities...")
        aspect_ids = set()
        aspect_names = {}
        aspect_mapping = {}
    
    # Step 3: Extract original KG1 and KG2 entity IDs (for query and original entity retrieval)
    print(f"\nStep 3: Extracting original entities from top-k results...")
    kg1_entity_ids, kg2_entity_ids = extract_entities_from_topk(topk_file_path)
    
    if len(kg1_entity_ids) == 0:
        print("Error: No KG1 entities found in top-k file.")
        return False
    
    if len(kg2_entity_ids) == 0:
        print("Error: No KG2 entities found in top-k file.")
        return False
    
    # Step 4: Create retrieval document (including original KG2 entities + aspect entities)
    print(f"\nStep 4: Creating retrieval document (original KG2 entities + aspect entities)...")
    
    # Create retrieval document (including original entities and aspect entities)
    written_count = create_retrieval_document_with_aspects(
        data_dir,
        kg2_entity_ids,
        aspect_ids,
        aspect_names,
        ent_ids_2_path,
        retrieval_doc_path
    )
    
    if written_count == 0:
        print("Error: Failed to create retrieval document.")
        return False
    
    # Step 5: Create query entities file (KG1 entities, only those in top-k)
    print(f"\nStep 5: Creating query entities file (KG1 entities from top-k)...")
    query_file = os.path.join(data_dir, "message_pool", "query_ent_ids_1.txt")
    query_count = create_query_entities_file(
        data_dir,
        kg1_entity_ids,
        ent_ids_1_path,
        query_file
    )
    
    if query_count == 0:
        print("Error: Failed to create query entities file.")
        return False
    
    print(f"  Query entities file saved: {query_file}")
    
    # Step 6: Backup original ent_ids_1 file and replace with query file
    original_ent_ids_1_backup = os.path.join(data_dir, "ent_ids_1.backup")
    use_temp_file = False
    
    try:
        # Backup original file
        if os.path.exists(ent_ids_1_path):
            import shutil
            shutil.copy2(ent_ids_1_path, original_ent_ids_1_backup)
            # Replace with query file
            shutil.copy2(query_file, ent_ids_1_path)
            use_temp_file = True
            print(f"  Temporarily replaced ent_ids_1 with query entities from top-k")
        
        # Step 7: Call neural_retrieval
        print(f"\nStep 7: Running neural retrieval...")
        print(f"  - Retrieval document: {retrieval_doc_path} ({written_count} entities)")
        print(f"    * Original KG2 entities: {len(kg2_entity_ids)}")
        print(f"    * Aspect entities: {len(aspect_ids)}")
        print(f"  - Query entities: {ent_ids_1_path} ({query_count} entities from top-k)")
        print(f"  - Query file saved: {query_file}")
        print(f"  - Output: {os.path.join(data_dir, 'message_pool', 'retriever_outputs.txt')}")
        if force_update:
            print(f"  - Force update: FAISS index will be regenerated\n")
        else:
            print()
        
        neural_retrieval(data_dir, force_rebuild_index=force_update)
        print("\n✓ Neural retrieval completed successfully!")
        
        # Step 8: Link retrieval results to original entities
        linked_output_file = os.path.join(data_dir, 'message_pool', 'retriever_outputs_linked.txt')
        if aspect_mapping:
            print(f"\nStep 8: Linking retrieval results to original entities...")
            link_retrieval_results_to_original(
                data_dir,
                aspect_mapping,
                os.path.join(data_dir, 'message_pool', 'retriever_outputs.txt'),
                linked_output_file
            )
        else:
            # If no aspect entities, directly copy retriever_outputs.txt as linked file
            import shutil
            retrieval_output_file = os.path.join(data_dir, 'message_pool', 'retriever_outputs.txt')
            if os.path.exists(retrieval_output_file):
                shutil.copy2(retrieval_output_file, linked_output_file)
                print(f"  No aspect entities, copied retriever_outputs.txt to retriever_outputs_linked.txt")
        
        # Step 9: Create multi-scale hypergraph representation folder
        print(f"\nStep 9: Creating multi-scale hypergraph representation folder...")
        create_multi_scale_hypergraph_representation(data_dir)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during neural retrieval: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original file
        if use_temp_file and os.path.exists(original_ent_ids_1_backup):
            import shutil
            shutil.move(original_ent_ids_1_backup, ent_ids_1_path)
            print(f"  Restored original ent_ids_1 file")
        
        # Query file saved to message_pool/query_ent_ids_1.txt, do not delete
        print(f"  Query entities file saved for future use: {query_file}")


def create_multi_scale_hypergraph_representation(data_dir):
    """
    Create multi-scale hypergraph representation folder
    
    Create multi-scale hypergraph representation folder under message_pool directory, containing:
    - L1 scale hypergraph: Simple-HHEA top-k results (integration_top_pair.txt)
    - L2 scale hypergraph: Retrieval results (retriever_outputs.txt)
    - L3 scale hypergraph: Linked retrieval results (retriever_outputs_linked.txt)
    
    Args:
        data_dir: Data directory path
    """
    message_pool_dir = os.path.join(data_dir, "message_pool")
    multi_scale_dir = os.path.join(message_pool_dir, "multi_scale_hypergraph")
    
    # Create multi-scale hypergraph representation folder
    os.makedirs(multi_scale_dir, exist_ok=True)
    
    # Source file paths
    source_files = {
        'L1': os.path.join(message_pool_dir, "integration_top_pair.txt"),
        'L2': os.path.join(message_pool_dir, "retriever_outputs.txt"),
        'L3': os.path.join(message_pool_dir, "retriever_outputs_linked.txt")
    }
    
    # Target file paths (renamed)
    target_files = {
        'L1': os.path.join(multi_scale_dir, "L1_hypergraph.txt"),
        'L2': os.path.join(multi_scale_dir, "L2_hypergraph.txt"),
        'L3': os.path.join(multi_scale_dir, "L3_hypergraph.txt")
    }
    
    created_files = []
    
    for scale in ['L1', 'L2', 'L3']:
        source_file = source_files[scale]
        target_file = target_files[scale]
        
        if os.path.exists(source_file):
            import shutil
            shutil.copy2(source_file, target_file)
            created_files.append((scale, target_file))
            print(f"  Created {scale} scale hypergraph: {target_file}")
        else:
            print(f"  Warning: Source file not found for {scale} scale: {source_file}")
    
    if created_files:
        print(f"  Multi-scale hypergraph representation created: {len(created_files)} scales")
        print(f"  Location: {multi_scale_dir}")
    else:
        print(f"  Warning: No multi-scale hypergraph files created")
    
    return created_files


def link_retrieval_results_to_original(data_dir, aspect_mapping, retrieval_output_file, linked_output_file):
    """
    Link aspect entities in retrieval results back to original entities
    
    Args:
        data_dir: Data directory
        aspect_mapping: Mapping from aspect to original entity {aspect_id: original_kg2_id}
        retrieval_output_file: Retrieval output file path
        linked_output_file: Linked output file path
    """
    if not os.path.exists(retrieval_output_file):
        print(f"Warning: Retrieval output file not found: {retrieval_output_file}")
        return
    
    linked_count = 0
    aspect_count = 0
    
    with open(retrieval_output_file, 'r', encoding='utf-8') as f_in, \
         open(linked_output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                kg1_id = parts[0]
                retrieved_id = int(parts[1])
                
                # Check if it's an aspect entity
                if retrieved_id in aspect_mapping:
                    # Link to original entity
                    original_id = aspect_mapping[retrieved_id]
                    f_out.write(f"{kg1_id}\t{original_id}\t{retrieved_id}\n")  # kg1_id, original_kg2_id, aspect_id
                    aspect_count += 1
                else:
                    # Original entity, output directly
                    f_out.write(f"{kg1_id}\t{retrieved_id}\n")
                
                linked_count += 1
    
    print(f"  Linked {linked_count} retrieval results:")
    print(f"    - Original entities: {linked_count - aspect_count}")
    print(f"    - Aspect entities (linked to original): {aspect_count}")
    print(f"  Linked results saved: {linked_output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Connect Simple-HHEA output and Neural Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python s4_to_retrieval.py --data_dir /path/to/data/icews_wiki
  
  # Specify dataset name
  python s4_to_retrieval.py --data_dir /path/to/data/icews_wiki --dataset icews_wiki
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory path (e.g., /path/to/data/icews_wiki)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (optional, extracted from data_dir path by default)"
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Execute conversion and retrieval
    success = s4_to_retrieval(args.data_dir, args.dataset)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ S4 to Retrieval process completed successfully!")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ S4 to Retrieval process failed!")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

