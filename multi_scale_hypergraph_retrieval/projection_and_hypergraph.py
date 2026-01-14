#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from collections import defaultdict
from typing import Set, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveEvolutionProjection:
    """
    Adaptive Symbolic Decoupling - Second phase of Adaptive Decoupling
    Implements paper Equation (1):
    - Temporal Projection: P̃_time(e^s, e^t) = M.(T(e^t), T(e^s))
      → Mask timestamps not in source entity
    - Modal Projection: P̃_modal(e^s, e^t) = M.(M(e^t), M(e^s))
      → Mask modality types not in source entity
    The projections implicitly construct modal hypergraphs H_m:
    - temporal_H/ → H_temporal nodes
    - text_H/    → H_text nodes
    - image_H/   → H_image nodes
    - audio_H/   → H_audio nodes
    - video_H/   → H_video nodes
    """
    def __init__(self, data_dir: str, output_dir: str):
        """
         Initialize

         Args:
             data_dir: Input data directory
             output_dir: Output result directory
         """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Data structures
        self.pre_alignment_pairs = []  # [(kg1_id, kg2_id), ...]
        self.modality_data = {
            'text_H': ({}, {}),      # (kg1_entities, kg2_entities)
            'image_H': ({}, {}),
            'audio_H': ({}, {}),
            'video_H': ({}, {})
        }
        self.timestamps = {}           # {timestamp_id: timestamp_value}
        self.entity_timestamps = ({}, {})  # (kg1_entity_times, kg2_entity_times)

    def load_data(self):

        logger.info("Starting to load data...")

        # 1. Load pre-alignment entity pairs
        self._load_pre_alignment_pairs()

        # 2. Load modality attributes
        self._load_modality_attributes()

        # 3. Load timestamps
        self._load_timestamps()

        # 4. Load triples and extract temporal information
        self._load_temporal_triples()

        logger.info("Data loading completed")

    def _load_pre_alignment_pairs(self):

        filepath = os.path.join(self.data_dir, 'message_pool', 'retriever_outputs.txt')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    kg1_id, kg2_id = int(parts[0]), int(parts[1])
                    self.pre_alignment_pairs.append((kg1_id, kg2_id))
        logger.info(f"Loaded pre-alignment entity pairs: {len(self.pre_alignment_pairs)} pairs")

    def _load_modality_attributes(self):
        """Load modality attributes"""
        modalities = {
            'text_H': ('txt_link_trans_1', 'txt_link_trans_2'),
            'image_H': ('image_link_trans_1', 'image_link_trans_2'),
            'audio_H': ('audio_link_trans_1', 'audio_link_trans_2'),
            'video_H': ('video_link_trans_1', 'video_link_trans_2')
        }

        for modality, (file1, file2) in modalities.items():
            # Load KG1
            kg1_data = self._load_modality_file(file1)
            self.modality_data[modality] = (kg1_data, {})

            # Load KG2
            kg2_data = self._load_modality_file(file2)
            self.modality_data[modality] = (kg1_data, kg2_data)

            logger.info(f"Loaded {modality} modality: KG1={len(kg1_data)}, KG2={len(kg2_data)}")

    def _load_modality_file(self, filename: str) -> Dict[int, str]:

        filepath = os.path.join(self.data_dir, filename)
        data = {}
        if not os.path.exists(filepath):
            logger.warning(f"Modality file not found: {filepath}, using empty data")
            return data
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = int(parts[0])
                    attribute = parts[1].strip()

                    if attribute:
                        data[entity_id] = attribute
        return data

    def _load_timestamps(self):

        filepath = os.path.join(self.data_dir, 'time_id')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    time_id = int(parts[0])
                    time_value = parts[1]
                    self.timestamps[time_id] = time_value
        logger.info(f"Loaded timestamps: {len(self.timestamps)} timestamps")

    def _load_temporal_triples(self):
        kg1_times = defaultdict(set)
        kg2_times = defaultdict(set)


        filepath = os.path.join(self.data_dir, 'triples_1')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    head_id = int(parts[0])
                    tail_id = int(parts[2])
                    time_start = int(parts[3])
                    time_end = int(parts[4])


                    for entity_id in [head_id, tail_id]:
                        for time_id in range(time_start, time_end + 1):
                            kg1_times[entity_id].add(time_id)


        filepath = os.path.join(self.data_dir, 'triples_2')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    head_id = int(parts[0])
                    tail_id = int(parts[2])
                    time_start = int(parts[3])
                    time_end = int(parts[4])

                    for entity_id in [head_id, tail_id]:
                        for time_id in range(time_start, time_end + 1):
                            kg2_times[entity_id].add(time_id)

        self.entity_timestamps = (dict(kg1_times), dict(kg2_times))
        logger.info(f"Extracted entity temporal information: KG1={len(kg1_times)}, KG2={len(kg2_times)}")

    def _get_common_modalities(self, kg1_id: int, kg2_id: int) -> Set[str]:
        """
        Apply Modal Mask: M.(M(e^t), M(e^s))
        Corresponds to P̃_modal in Equation (1):
        Removes modality types absent in source entity
        Returns:
            Set of modalities passing the mask
        """
        ...
        common = set()
        for modality in ['text_H', 'image_H', 'audio_H', 'video_H']:
            kg1_data, kg2_data = self.modality_data[modality]
            if kg1_id in kg1_data and kg2_id in kg2_data:
                common.add(modality)
        return common

    def _has_temporal_overlap(self, kg1_id: int, kg2_id: int) -> bool:
        """
        Apply Temporal Mask: M.(T(e^t), T(e^s))
        Corresponds to P̃_time in Equation (1):
        Checks if target entity has timestamps in source entity's range
        Returns:
            True if temporal_H overlap exists (passes mask)
        """
        kg1_times, kg2_times = self.entity_timestamps

        times1 = kg1_times.get(kg1_id, set())
        times2 = kg2_times.get(kg2_id, set())


        return len(times1 & times2) > 0

    def project_and_save(self, batch_size: int = 10000):
        """
        Execute projection and save by category

        Args:
            batch_size: Batch processing size
        """
        logger.info("Starting adaptive evolution projection...")


        os.makedirs(self.output_dir, exist_ok=True)
        subdirs = ['temporal_H', 'text_H', 'image_H', 'audio_H', 'video_H',
                   'text_temporal_H', 'image_temporal_H', 'audio_temporal_H', 'video_temporal_H']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)


        stats = defaultdict(int)


        total_pairs = len(self.pre_alignment_pairs)
        batch_id = 0

        for i in range(0, total_pairs, batch_size):
            batch = self.pre_alignment_pairs[i:i + batch_size]
            batch_results = self._process_batch(batch, stats)


            self._save_batch_results(batch_results, batch_id)

            batch_id += 1
            if (i + batch_size) % 50000 == 0:
                logger.info(f"Process: {min(i + batch_size, total_pairs)}/{total_pairs}")


        self._print_statistics(stats)
        logger.info("Projection finish！")

    def _process_batch(self, batch: List[Tuple[int, int]], stats: Dict) -> Dict:

        results = {
            'temporal_H': [],
            'text_H': [],
            'image_H': [],
            'audio_H': [],
            'video_H': [],
            'text_temporal_H': [],
            'image_temporal_H': [],
            'audio_temporal_H': [],
            'video_temporal_H': []
        }

        for kg1_id, kg2_id in batch:
            # 1. Check common modalities
            common_modalities = self._get_common_modalities(kg1_id, kg2_id)

            # 2. Check temporal overlap
            has_temporal = self._has_temporal_overlap(kg1_id, kg2_id)

            # 3. Classify and save
            pair_str = f"{kg1_id}\t{kg2_id}"

            # Pure temporal projection
            if has_temporal:
                results['temporal_H'].append(pair_str)
                stats['temporal_H'] += 1

            # Pure modal projection
            for modality in common_modalities:
                results[modality].append(pair_str)
                stats[modality] += 1

            # Modal + temporal projection (both satisfied)
            if has_temporal:
                for modality in common_modalities:
                    key = f"{modality}_temporal"
                    keys = {
                        'temporal_H': 'temporal_H',
                        'text_H': 'text_H',
                        'image_H': 'image_H',
                        'audio_H': 'audio_H',
                        'video_H': 'video_H',
                        'text_H_temporal': 'text_temporal_H',
                        'image_H_temporal': 'image_temporal_H',
                        'audio_H_temporal': 'audio_temporal_H',
                        'video_H_temporal': 'video_temporal_H'
                    }
                    results[keys[key]].append(pair_str)
                    stats[key] += 1

        return results

    def _save_batch_results(self, batch_results: Dict, batch_id: int):
        for category, pairs in batch_results.items():
            if not pairs:
                continue

            output_path = os.path.join(self.output_dir, category,
                                      f'projection_pairs_batch_{batch_id:04d}.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(pairs) + '\n')

    def _print_statistics(self, stats: Dict):
        logger.info("\n" + "="*50)
        logger.info("Projection info:")
        logger.info("="*50)
        for category, count in sorted(stats.items()):
            logger.info(f"{category:20s}: {count:8d} pairs")
        logger.info("="*50)


def merge_batch_files(output_dir: str):
    """
    Merge batch files under each category (optional)

    Args:
        output_dir: Output directory
    """
    logger.info("Starting to merge batch files...")

    subdirs = ['temporal_H', 'text_H', 'image_H', 'audio_H', 'video_H',
               'text_temporal_H', 'image_temporal_H', 'audio_temporal_H', 'video_temporal_H']

    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        batch_files = sorted([f for f in os.listdir(subdir_path) if f.startswith('projection_pairs_batch_')])

        if not batch_files:
            continue

        merged_file = os.path.join(output_dir, subdir, 'projection_pairs_merged.txt')
        with open(merged_file, 'w', encoding='utf-8') as outf:
            for batch_file in batch_files:
                with open(os.path.join(subdir_path, batch_file), 'r', encoding='utf-8') as inf:
                    outf.write(inf.read())

        logger.info(f"Merged {subdir}: {len(batch_files)} batch files")

def get_hypergraph_statistics(self, output_dir: str) -> Dict:
    """
    Get statistics of implicitly constructed hypergraphs
    Returns:
        Dict: Number of nodes and hyperedges for each modal hypergraph
        {
            'H_temporal': {'nodes': 1000, 'hyperedges': 500},
            'H_text': {'nodes': 800, 'hyperedges': 400},
            ...
        }
    """
    stats = {}
    for category in ['temporal_H', 'text_H', 'image_H', 'audio_H', 'video_H']:
        file_path = os.path.join(output_dir, category, 'entity_pairs.txt')
        if os.path.exists(file_path):
            pairs = self._load_pairs(file_path)
            nodes = set()
            for kg1_id, kg2_id in pairs:
                nodes.add(kg1_id)
                nodes.add(kg2_id)
            stats[f'H_{category}'] = {
                'nodes': len(nodes),
                'hyperedges': len(pairs)  # Each source entity corresponds to a hyperedge
            }
    return stats

if __name__ == "__main__":
    # Configure paths
    DATA_DIR = "/home/dex/Desktop/entity_sy/HyDRA/data/icews_wiki"          # Input data directory
    OUTPUT_DIR = "/home/dex/Desktop/entity_sy/HyDRA/data/icews_wiki/message_pool"  # Output directory

    # Create processor instance
    processor = AdaptiveEvolutionProjection(DATA_DIR, OUTPUT_DIR)

    # Load data
    processor.load_data()

    # Execute projection and save (batch size is adjustable)
    processor.project_and_save(batch_size=10000)

    # Optional: merge batch files
    merge_batch_files(OUTPUT_DIR)

    logger.info("All processing completed!")