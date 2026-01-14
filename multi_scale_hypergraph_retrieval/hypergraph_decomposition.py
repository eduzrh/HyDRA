#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypergraph Decomposition Module

Features:
1. Decompose entity pairs from S4's top-k output into multiple aspects of KG2 entities
2. Two decomposition methods:
   - Temporal projection: Based on common time intervals of entity pairs, form temporal aspect KG2 entities
   - Relational projection: Based on common relations + tail entities, form relational aspect KG2 entities
3. Generate aspect entities for retrieval and maintain mapping from aspects to original entities

References:
- Projection hypergraph construction module in the paper
- Multi-modal decomposition approach in MTKGA-Wild
"""

import os
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HypergraphDecomposition:
    """
    Hypergraph Decomposition Class
    
    Decomposes entity pairs into multiple aspects of KG2 entities:
    1. Temporal aspects: Based on common time intervals
    2. Relational aspects: Based on common relations + tail entities
    """
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize
        
        Args:
            data_dir: Data directory path
            output_dir: Output directory path (default: data_dir/message_pool)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, "message_pool")
        
        # Data structures
        self.entity_pairs = []  # [(kg1_id, kg2_id), ...] loaded from top-k file
        self.kg1_triples = []  # [(head, rel, tail, time_start, time_end), ...]
        self.kg2_triples = []  # [(head, rel, tail, time_start, time_end), ...]
        self.kg1_entity_times = defaultdict(set)  # {entity_id: {time_id, ...}}
        self.kg2_entity_times = defaultdict(set)  # {entity_id: {time_id, ...}}
        self.kg1_entity_relations = defaultdict(set)  # {entity_id: {(rel_id, tail_id), ...}}
        self.kg2_entity_relations = defaultdict(set)  # {entity_id: {(rel_id, tail_id), ...}}
        
        # Relation alignment mapping {kg1_rel_id: {kg2_rel_id, ...}} supports one-to-many
        self.relation_alignment = defaultdict(set)  # {kg1_rel_id: {kg2_rel_id, ...}}
        
        # Aspect entity mappings
        self.temporal_aspects = {}  # {aspect_id: (original_kg2_id, [(time_start, time_end), ...])}
        self.relational_aspects = {}  # {aspect_id: (original_kg2_id, {(rel_id, tail_id), ...})}
        self.aspect_to_original = {}  # {aspect_id: original_kg2_id} for linking retrieval results
        
        # Aspect entity counters
        self.temporal_aspect_counter = 1000000  # Start from large number to avoid conflicts with real entity IDs
        self.relational_aspect_counter = 2000000
        
    def load_data(self):
        """Load necessary data"""
        logger.info("Loading data for hypergraph decomposition...")
        
        # 1. Load entity pairs (from top-k file)
        self._load_entity_pairs()
        
        # 2. Load triples and temporal information
        self._load_triples()
        
        # 3. Extract entity temporal information
        self._extract_entity_temporal_info()
        
        # 4. Extract entity relation information
        self._extract_entity_relation_info()
        
        # 5. Load relation alignment information (if exists)
        self._load_relation_alignment()
        
        logger.info("Data loading completed")
        logger.info(f"  Entity pairs: {len(self.entity_pairs)}")
        logger.info(f"  KG1 triples: {len(self.kg1_triples)}")
        logger.info(f"  KG2 triples: {len(self.kg2_triples)}")
        logger.info(f"  Relation alignments: {len(self.relation_alignment)}")
        
    def _load_entity_pairs(self):
        """Load entity pairs from top-k file"""
        topk_file = os.path.join(self.data_dir, "message_pool", "integration_top_pair.txt")
        
        if not os.path.exists(topk_file):
            logger.warning(f"Top-k file not found: {topk_file}")
            return
        
        with open(topk_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        kg1_id = int(parts[0].strip())
                        kg2_id = int(parts[1].strip())
                        self.entity_pairs.append((kg1_id, kg2_id))
                    except (ValueError, IndexError):
                        continue
        
        logger.info(f"Loaded {len(self.entity_pairs)} entity pairs from top-k file")
    
    def _load_triples(self):
        """Load triples data"""
        # Load KG1 triples
        triples_1_path = os.path.join(self.data_dir, "triples_1")
        if os.path.exists(triples_1_path):
            with open(triples_1_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head = int(parts[0])
                        rel = int(parts[1])
                        tail = int(parts[2])
                        time_start = int(parts[3]) if len(parts) > 3 else 0
                        time_end = int(parts[4]) if len(parts) > 4 else time_start
                        self.kg1_triples.append((head, rel, tail, time_start, time_end))
        
        # Load KG2 triples
        triples_2_path = os.path.join(self.data_dir, "triples_2")
        if os.path.exists(triples_2_path):
            with open(triples_2_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head = int(parts[0])
                        rel = int(parts[1])
                        tail = int(parts[2])
                        time_start = int(parts[3]) if len(parts) > 3 else 0
                        time_end = int(parts[4]) if len(parts) > 4 else time_start
                        self.kg2_triples.append((head, rel, tail, time_start, time_end))
        
        logger.info(f"Loaded KG1 triples: {len(self.kg1_triples)}")
        logger.info(f"Loaded KG2 triples: {len(self.kg2_triples)}")
    
    def _extract_entity_temporal_info(self):
        """Extract entity temporal information"""
        # KG1 entity times
        for head, rel, tail, time_start, time_end in self.kg1_triples:
            for entity_id in [head, tail]:
                for time_id in range(time_start, time_end + 1):
                    self.kg1_entity_times[entity_id].add(time_id)
        
        # KG2 entity times
        for head, rel, tail, time_start, time_end in self.kg2_triples:
            for entity_id in [head, tail]:
                for time_id in range(time_start, time_end + 1):
                    self.kg2_entity_times[entity_id].add(time_id)
        
        logger.info(f"Extracted temporal info: KG1={len(self.kg1_entity_times)}, KG2={len(self.kg2_entity_times)}")
    
    def _extract_entity_relation_info(self):
        """Extract entity relation information (relations + tail entities when entity is head)"""
        # KG1 entity relations
        for head, rel, tail, time_start, time_end in self.kg1_triples:
            self.kg1_entity_relations[head].add((rel, tail))
        
        # KG2 entity relations
        for head, rel, tail, time_start, time_end in self.kg2_triples:
            self.kg2_entity_relations[head].add((rel, tail))
        
        logger.info(f"Extracted relation info: KG1={len(self.kg1_entity_relations)}, KG2={len(self.kg2_entity_relations)}")
    
    def _get_common_time_intervals(self, kg1_id: int, kg2_id: int) -> List[Tuple[int, int]]:
        """
        Get common time intervals for entity pair
        
        Args:
            kg1_id: KG1 entity ID
            kg2_id: KG2 entity ID
            
        Returns:
            List of (time_start, time_end) tuples representing common intervals
        """
        times1 = self.kg1_entity_times.get(kg1_id, set())
        times2 = self.kg2_entity_times.get(kg2_id, set())
        
        common_times = times1 & times2
        
        if not common_times:
            return []
        
        # Merge common time points into continuous intervals
        sorted_times = sorted(common_times)
        intervals = []
        
        if not sorted_times:
            return intervals
        
        start = sorted_times[0]
        end = sorted_times[0]
        
        for time_id in sorted_times[1:]:
            if time_id == end + 1:
                end = time_id
            else:
                intervals.append((start, end))
                start = time_id
                end = time_id
        
        intervals.append((start, end))
        
        return intervals
    
    def _load_relation_alignment(self):
        """Load relation alignment information"""
        alignment_file = os.path.join(self.output_dir, "relation_alignment.txt")
        if not os.path.exists(alignment_file):
            logger.warning(f"Relation alignment file not found: {alignment_file}")
            logger.warning("Will use exact relation ID matching (no alignment)")
            return
        
        with open(alignment_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header or empty file
                logger.warning("Relation alignment file is empty")
                return
            
            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        kg1_rel_id = int(parts[0])
                        kg2_rel_id = int(parts[2])
                        self.relation_alignment[kg1_rel_id].add(kg2_rel_id)
                    except ValueError:
                        continue
        
        logger.info(f"Loaded {len(self.relation_alignment)} relation alignments")
        # Count one-to-many cases
        one_to_many = sum(1 for v in self.relation_alignment.values() if len(v) > 1)
        if one_to_many > 0:
            logger.info(f"  One-to-many alignments: {one_to_many}")
    
    def _get_common_relations(self, kg1_id: int, kg2_id: int) -> Set[Tuple[int, int]]:
        """
        Get common relations + tail entities for entity pair (using relation alignment information)
        
        Logic:
        1. Get all relations of KG1 entity
        2. Through relation alignment, find corresponding KG2 relations
        3. Check if KG2 entity has these aligned relations
        4. If KG2 has aligned relations, collect all combinations of relation + tail entity
        
        Args:
            kg1_id: KG1 entity ID
            kg2_id: KG2 entity ID
            
        Returns:
            Set of (kg2_rel_id, kg2_tail_id) tuples (using KG2 relation IDs and tail entity IDs)
        """
        kg1_rels = self.kg1_entity_relations.get(kg1_id, set())  # {(kg1_rel_id, tail_id), ...}
        kg2_rels = self.kg2_entity_relations.get(kg2_id, set())  # {(kg2_rel_id, tail_id), ...}
        
        common_rels = set()
        
        if not self.relation_alignment:
            # If no relation alignment information, return empty set
            return common_rels
        
        # Use relation alignment information
        # Collect KG2 relation sets corresponding to all relations of KG1 entity
        kg1_rel_ids = {rel_id for rel_id, _ in kg1_rels}
        kg2_aligned_rel_ids = set()
        
        for kg1_rel_id in kg1_rel_ids:
            # Get KG2 relation set corresponding to KG1 relation (supports one-to-many)
            aligned_kg2_rels = self.relation_alignment.get(kg1_rel_id, set())
            kg2_aligned_rel_ids.update(aligned_kg2_rels)
        
        # Check if KG2 entity has these aligned relations, if yes, collect all relation + tail entity combinations
        for kg2_rel_id, kg2_tail_id in kg2_rels:
            if kg2_rel_id in kg2_aligned_rel_ids:
                common_rels.add((kg2_rel_id, kg2_tail_id))
        
        return common_rels
    
    def decompose(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Execute hypergraph decomposition
        
        For each entity pair, generate:
        - A temporal aspect: containing all common time intervals
        - A relational aspect: containing all common relation + tail entity combinations
        
        Returns:
            Dict containing:
            - 'temporal_aspects': [(kg1_id, temporal_aspect_id), ...]
            - 'relational_aspects': [(kg1_id, relational_aspect_id), ...]
            - 'aspect_mapping': {aspect_id: original_kg2_id}
        """
        logger.info("Starting hypergraph decomposition...")
        
        temporal_aspects = []
        relational_aspects = []
        
        for kg1_id, kg2_id in self.entity_pairs:
            # 1. Temporal projection: Generate a temporal aspect containing all common time intervals
            common_intervals = self._get_common_time_intervals(kg1_id, kg2_id)
            if common_intervals:
                aspect_id = self.temporal_aspect_counter
                self.temporal_aspect_counter += 1
                
                # Store all common time intervals
                self.temporal_aspects[aspect_id] = (kg2_id, common_intervals)
                self.aspect_to_original[aspect_id] = kg2_id
                temporal_aspects.append((kg1_id, aspect_id))
            
            # 2. Relational projection: Generate a relational aspect containing all common relation + tail entity combinations
            common_rels = self._get_common_relations(kg1_id, kg2_id)
            if common_rels:
                aspect_id = self.relational_aspect_counter
                self.relational_aspect_counter += 1
                
                # Store all common relation + tail entity combinations
                self.relational_aspects[aspect_id] = (kg2_id, common_rels)
                self.aspect_to_original[aspect_id] = kg2_id
                relational_aspects.append((kg1_id, aspect_id))
        
        logger.info(f"Generated temporal aspects: {len(temporal_aspects)}")
        logger.info(f"Generated relational aspects: {len(relational_aspects)}")
        
        return {
            'temporal_aspects': temporal_aspects,
            'relational_aspects': relational_aspects,
            'aspect_mapping': self.aspect_to_original
        }
    
    def save_decomposed_aspects(self, decomposition_results: Dict):
        """
        Save decomposed aspect entities
        
        Args:
            decomposition_results: Return value of decompose() method
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Save temporal aspect entity pairs
        temporal_file = os.path.join(self.output_dir, "temporal_aspect_pairs.txt")
        with open(temporal_file, 'w', encoding='utf-8') as f:
            for kg1_id, aspect_id in decomposition_results['temporal_aspects']:
                f.write(f"{kg1_id}\t{aspect_id}\n")
        logger.info(f"Saved temporal aspect pairs: {temporal_file}")
        
        # 2. Save relational aspect entity pairs
        relational_file = os.path.join(self.output_dir, "relational_aspect_pairs.txt")
        with open(relational_file, 'w', encoding='utf-8') as f:
            for kg1_id, aspect_id in decomposition_results['relational_aspects']:
                f.write(f"{kg1_id}\t{aspect_id}\n")
        logger.info(f"Saved relational aspect pairs: {relational_file}")
        
        # 3. Save aspect mapping (for linking retrieval results)
        mapping_file = os.path.join(self.output_dir, "aspect_to_original_mapping.txt")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            for aspect_id, original_id in decomposition_results['aspect_mapping'].items():
                f.write(f"{aspect_id}\t{original_id}\n")
        logger.info(f"Saved aspect mapping: {mapping_file}")
        
        # 4. Save temporal aspect detailed information (containing all common time intervals)
        temporal_info_file = os.path.join(self.output_dir, "temporal_aspect_info.txt")
        with open(temporal_info_file, 'w', encoding='utf-8') as f:
            for aspect_id, (original_id, intervals) in self.temporal_aspects.items():
                # Serialize time interval list to string
                intervals_str = ";".join([f"{start}-{end}" for start, end in intervals])
                f.write(f"{aspect_id}\t{original_id}\t{intervals_str}\n")
        logger.info(f"Saved temporal aspect info: {temporal_info_file}")
        
        # 5. Save relational aspect detailed information (containing all common relation + tail entity combinations)
        relational_info_file = os.path.join(self.output_dir, "relational_aspect_info.txt")
        with open(relational_info_file, 'w', encoding='utf-8') as f:
            for aspect_id, (original_id, common_rels) in self.relational_aspects.items():
                # Serialize relation combination list to string
                rels_str = ";".join([f"{rel_id},{tail_id}" for rel_id, tail_id in common_rels])
                f.write(f"{aspect_id}\t{original_id}\t{rels_str}\n")
        logger.info(f"Saved relational aspect info: {relational_info_file}")
    
    def create_aspect_entities_for_retrieval(self, decomposition_results: Dict) -> Set[int]:
        """
        Create aspect entity set for retrieval
        
        Args:
            decomposition_results: Return value of decompose() method
            
        Returns:
            Set of aspect entity IDs (temporal + relational)
        """
        all_aspect_ids = set()
        
        # Collect all aspect entity IDs
        for _, aspect_id in decomposition_results['temporal_aspects']:
            all_aspect_ids.add(aspect_id)
        
        for _, aspect_id in decomposition_results['relational_aspects']:
            all_aspect_ids.add(aspect_id)
        
        logger.info(f"Total aspect entities for retrieval: {len(all_aspect_ids)}")
        return all_aspect_ids
    
    def create_aspect_entity_names(self, aspect_ids: Set[int], ent_ids_2_path: str, 
                                   rel_ids_2_path: str = None, ent_ids_2_dict: Dict[int, str] = None) -> Dict[int, str]:
        """
        Create names for aspect entities (based on original entity name + aspect information)
        
        Args:
            aspect_ids: Set of aspect entity IDs
            ent_ids_2_path: KG2 entity file path
            rel_ids_2_path: KG2 relation file path (optional, for getting relation names)
            ent_ids_2_dict: KG2 entity dictionary (optional, can pass directly if already loaded)
            
        Returns:
            Dict: {aspect_id: aspect_entity_name}
        """
        # Load original entity names
        if ent_ids_2_dict is None:
            original_entity_names = {}
            if os.path.exists(ent_ids_2_path):
                with open(ent_ids_2_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            entity_id = int(parts[0])
                            entity_name = parts[1]
                            original_entity_names[entity_id] = entity_name
        else:
            original_entity_names = ent_ids_2_dict
        
        # Load relation names (optional)
        relation_names = {}
        if rel_ids_2_path and os.path.exists(rel_ids_2_path):
            with open(rel_ids_2_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        rel_id = int(parts[0])
                        rel_name = parts[1]
                        relation_names[rel_id] = rel_name
        
        aspect_names = {}
        
        # Temporal aspect names: original name + all common time intervals
        for aspect_id, (original_id, intervals) in self.temporal_aspects.items():
            if aspect_id in aspect_ids:
                original_name = original_entity_names.get(original_id, f"Entity_{original_id}")
                # Format time interval list
                time_strs = []
                for time_start, time_end in intervals:
                    if time_start == time_end:
                        time_strs.append(str(time_start))
                    else:
                        time_strs.append(f"{time_start}-{time_end}")
                time_str = ",".join(time_strs)
                aspect_name = f"{original_name}_[T:{time_str}]"
                aspect_names[aspect_id] = aspect_name
        
        # Relational aspect names: original name + all common relation + tail entity combinations
        for aspect_id, (original_id, common_rels) in self.relational_aspects.items():
            if aspect_id in aspect_ids:
                original_name = original_entity_names.get(original_id, f"Entity_{original_id}")
                # Format relation combination list
                rel_strs = []
                for rel_id, tail_id in sorted(common_rels):  # Sort for consistency
                    rel_name = relation_names.get(rel_id, f"R{rel_id}")
                    tail_name = original_entity_names.get(tail_id, f"E{tail_id}")
                    rel_strs.append(f"{rel_name}→{tail_name}")
                rel_str = ",".join(rel_strs)
                aspect_name = f"{original_name}_[R:{rel_str}]"
                aspect_names[aspect_id] = aspect_name
        
        logger.info(f"Created names for {len(aspect_names)} aspect entities")
        return aspect_names


def run_hypergraph_decomposition(data_dir: str, output_dir: str = None):
    """
    Run hypergraph decomposition pipeline
    
    Args:
        data_dir: Data directory path
        output_dir: Output directory path (optional)
        
    Returns:
        Dict: Decomposition results, containing aspect entity sets and mapping information
    """
    logger.info("=" * 80)
    logger.info("Hypergraph Decomposition")
    logger.info("=" * 80)
    
    decomposer = HypergraphDecomposition(data_dir, output_dir)
    decomposer.load_data()
    
    # Execute decomposition
    decomposition_results = decomposer.decompose()
    
    # Save results
    decomposer.save_decomposed_aspects(decomposition_results)
    
    # Create aspect entity set for retrieval
    aspect_ids = decomposer.create_aspect_entities_for_retrieval(decomposition_results)
    
    # Create aspect entity names
    ent_ids_2_path = os.path.join(data_dir, "ent_ids_2")
    rel_ids_2_path = os.path.join(data_dir, "rel_ids_2")
    aspect_names = decomposer.create_aspect_entity_names(aspect_ids, ent_ids_2_path, rel_ids_2_path)
    
    logger.info("=" * 80)
    logger.info("Hypergraph Decomposition Completed")
    logger.info("=" * 80)
    
    return {
        'aspect_ids': aspect_ids,
        'aspect_names': aspect_names,
        'aspect_mapping': decomposition_results['aspect_mapping'],
        'temporal_aspects': decomposition_results['temporal_aspects'],
        'relational_aspects': decomposition_results['relational_aspects'],
        'decomposer': decomposer  # Keep decomposer instance for subsequent linking
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hypergraph Decomposition")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path (optional)")
    
    args = parser.parse_args()
    
    run_hypergraph_decomposition(args.data_dir, args.output_dir)

