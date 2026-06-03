#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDRA: Main Pipeline Script

Stages:
1. Encoding and Integration
2. Scale-Adaptive Entity Projection
3. Multi-Scale Fusion

Usage:
    python HyDRA_main.py --data_dir <dataset_name_or_path> [options]

Supported datasets:
    WildBETA, BETA, icews_wiki, icews_yago, YAGO-WIKI50K-1K,
    DICEWS-200, en_fr, DBP-WIKI, DOREMUS, AGROLD
"""

import os
import sys
import argparse
import subprocess
from ablation_config import AblationConfig


def check_s4_output(data_dir):
    """Check if Encoding and Integration output file exists."""
    s4_output_file = os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')
    return os.path.exists(s4_output_file) and os.path.getsize(s4_output_file) > 0


def count_unique_kg1_entities(data_dir):
    """Count unique KG1 entities in integration_top_pair.txt."""
    s4_output_file = os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')
    
    if not os.path.exists(s4_output_file):
        return 0
    
    kg1_entities = set()
    try:
        with open(s4_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        kg1_id = int(parts[0])
                        kg1_entities.add(kg1_id)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Error reading encoding and integration output file: {e}")
        return 0
    
    return len(kg1_entities)


def ensure_ent_ids_1_restored(data_dir):
    """Ensure ent_ids_1 file is restored if it was temporarily replaced."""
    ent_ids_1_path = os.path.join(data_dir, "ent_ids_1")
    backup_path = os.path.join(data_dir, "ent_ids_1.backup")
    
    if os.path.exists(backup_path):
        import shutil
        print(f"  Warning: Found backup file, restoring ent_ids_1 from backup...")
        shutil.move(backup_path, ent_ids_1_path)
        print(f"  Restored ent_ids_1 from backup")
    
    if os.path.exists(ent_ids_1_path):
        with open(ent_ids_1_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
            if line_count < 1000:
                print(f"  Warning: ent_ids_1 has only {line_count} lines, which seems too small")
                return False
        return True
    else:
        print(f"  Error: ent_ids_1 file not found: {ent_ids_1_path}")
        return False


def run_s4_training(data_dir, cuda=0, epochs=500, skip_encoding_integration=False,
                    multi_granularity_time=False, add_noise=False, noise_ratio=0.0,
                    ablation_config=None):
    """Run Stage 1: Encoding and Integration."""
    print("\n" + "=" * 80)
    print("Stage 1: Encoding and Integration")
    print("=" * 80 + "\n")
    
    if skip_encoding_integration:
        print("  [SKIP] encoding_and_integration step is skipped")
        s4_output_file = os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')
        if os.path.exists(s4_output_file) and os.path.getsize(s4_output_file) > 0:
            print(f"  ✓ Encoding and Integration output file already exists")
            return True
        else:
            print(f"  ⚠ Warning: Encoding and Integration output file not found")
            return True
    
    print("Checking ent_ids_1 file before encoding and integration...")
    if not ensure_ent_ids_1_restored(data_dir):
        print("  Error: ent_ids_1 file is not in correct state.")
        return False
    
    s4_script = os.path.join(os.path.dirname(__file__), 'encoding_and_integration', 'run_s4_standalone.py')
    
    if os.path.exists(s4_script):
        try:
            cmd = [
                sys.executable,
                s4_script,
                '--data_dir', data_dir,
                '--cuda', str(cuda),
                '--epochs', str(epochs)
            ]
            
            if ablation_config is None:
                use_multi_gran = multi_granularity_time
            else:
                use_multi_gran = multi_granularity_time and ablation_config.use_multi_granular_temporal_encoder
            
            if use_multi_gran:
                cmd.append('--multi_granularity_time')
            
            cmd.extend(['--noise_ratio', str(noise_ratio)])
            if add_noise:
                cmd.append('--add_noise')
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running Encoding and Integration: {e}")
            return False
    else:
        print(f"Warning: {s4_script} not found.")
        return False


def run_s4_to_retrieval(data_dir, iteration=1, ablation_config=None):
    """Run Stage 2: Scale-Adaptive Entity Projection."""
    print("\n" + "=" * 80)
    print("Stage 2: Scale-Adaptive Entity Projection")
    print("=" * 80 + "\n")
    
    try:
        from scale_adaptive_entity_projection.entity_projection import s4_to_retrieval
        success = s4_to_retrieval(data_dir, iteration=iteration, ablation_config=ablation_config)
        return success
    except ImportError as e:
        print(f"Error importing s4_to_retrieval: {e}")
        return False
    except Exception as e:
        print(f"Error running s4_to_retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_multi_scale_fusion(data_dir, ablation_config=None, snapshot_sup_pairs=False, iteration=0):
    """Run Stage 4: Multi-Scale Fusion."""
    print("\n" + "=" * 80)
    print("Stage 4: Multi-Scale Fusion")
    print("=" * 80 + "\n")
    
    try:
        from multi_scale_fusion.multi_scale_fusion import multi_scale_fusion
        aligned_pairs = multi_scale_fusion(
            data_dir, ablation_config=ablation_config,
            snapshot_sup_pairs=snapshot_sup_pairs, iteration=iteration
        )
        return aligned_pairs is not None and len(aligned_pairs) > 0
    except ImportError as e:
        print(f"Error importing multi_scale_fusion: {e}")
        return False
    except Exception as e:
        print(f"Error running multi_scale_fusion: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline(data_dir, skip_s4=False, only_s4=False, cuda=0, epochs=500, max_iterations=3, min_kg1_entities=50, skip_encoding_integration=False, multi_granularity_time=False, add_noise=False, noise_ratio=0.0, ablation_config=None, snapshot_sup_pairs=False):
    """Run the complete HyDRA pipeline with iteration control and ablation support."""
    if ablation_config is None:
        ablation_config = AblationConfig()
    
    print("\n" + "=" * 80)
    print("HyDRA: Complete Pipeline")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Max iterations: {max_iterations}")
    print(f"Min KG1 entities threshold: {min_kg1_entities}")
    print(f"Name-embedding noise (Sec. 5.6): add_noise={add_noise}, noise_ratio={noise_ratio}")
    print(f"Ablation Config: {ablation_config.get_description()}")
    print("=" * 80 + "\n")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return False
    
    if only_s4:
        success_steps = []
        if not skip_s4:
            if check_s4_output(data_dir):
                print("✓ Encoding and Integration output file already exists")
                success_steps.append("Encoding and Integration (already exists)")
            else:
                if run_s4_training(data_dir, cuda=cuda, epochs=epochs, skip_encoding_integration=skip_encoding_integration, multi_granularity_time=multi_granularity_time, add_noise=add_noise, noise_ratio=noise_ratio, ablation_config=ablation_config):
                    success_steps.append("Encoding and Integration")
                else:
                    print("✗ Encoding and Integration failed")
                    return False
        else:
            if not check_s4_output(data_dir):
                print("✗ Error: Encoding and Integration output file not found")
                return False
            else:
                print("✓ Encoding and Integration output file exists")
                success_steps.append("Encoding and Integration (skipped)")
        
        print("\n" + "=" * 80)
        print("✓ Stage 1: Encoding and Integration completed")
        print("=" * 80 + "\n")
        return True
    
    iteration = 0
    all_success_steps = []
    
    while iteration < max_iterations:
        iteration += 1
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{max_iterations}")
        print("=" * 80 + "\n")
        
        success_steps = []
        
        if skip_s4 and iteration == 1:
            if not check_s4_output(data_dir):
                print("✗ Error: Encoding and Integration output file not found")
                return False
            else:
                print("✓ Encoding and Integration output file exists, skipping (iteration 1 only)")
                success_steps.append("Encoding and Integration (skipped in iteration 1)")
        else:
            if iteration > 1:
                print(f"  Re-running Encoding and Integration for iteration {iteration}...")
            if run_s4_training(data_dir, cuda=cuda, epochs=epochs, skip_encoding_integration=skip_encoding_integration, multi_granularity_time=multi_granularity_time, add_noise=add_noise, noise_ratio=noise_ratio, ablation_config=ablation_config):
                success_steps.append("Encoding and Integration")
            else:
                print("✗ Encoding and Integration failed")
                if iteration == 1:
                    return False
                print("  Continuing with existing files...")
        
        kg1_count = count_unique_kg1_entities(data_dir)
        print(f"\nUnique KG1 entities: {kg1_count}")
        
        if kg1_count < min_kg1_entities:
            print(f"\n{'=' * 80}")
            print(f"Stopping: KG1 entities ({kg1_count}) < threshold ({min_kg1_entities})")
            print(f"{'=' * 80}\n")
            all_success_steps.extend(success_steps)
            break
        
        if run_s4_to_retrieval(data_dir, iteration=iteration, ablation_config=ablation_config):
            success_steps.append("Scale-Adaptive Entity Projection")
        else:
            print("✗ Scale-Adaptive Entity Projection failed")
            if iteration == 1:
                return False
            continue
        
        if run_multi_scale_fusion(data_dir, ablation_config=ablation_config,
                                  snapshot_sup_pairs=snapshot_sup_pairs, iteration=iteration):
            success_steps.append("Multi-Scale Fusion")
        else:
            print("✗ Multi-Scale Fusion failed")
            if iteration == 1:
                return False
            continue
        
        all_success_steps.extend(success_steps)
        
        if iteration >= max_iterations:
            print(f"\n{'=' * 80}")
            print(f"Reached maximum iterations: {max_iterations}")
            print(f"{'=' * 80}\n")
            break
    
    print("\n" + "=" * 80)
    print("HyDRA Pipeline Summary")
    print("=" * 80)
    print(f"Total iterations: {iteration}/{max_iterations}")
    print(f"Final KG1 entities count: {count_unique_kg1_entities(data_dir)}")
    print(f"Completed steps: {', '.join(all_success_steps)}")
    print("=" * 80 + "\n")
    
    return True


def normalize_data_dir(data_dir):
    """Normalize data directory path: convert dataset name to data/{name} if needed."""
    supported_datasets = [
        'WildBETA', 'BETA', 'icews_wiki', 'icews_yago', 'YAGO-WIKI50K-1K',
        'DICEWS-200', 'en_fr', 'DBP-WIKI', 'DOREMUS', 'AGROLD'
    ]
    
    if os.path.isabs(data_dir):
        return data_dir
    
    if os.sep in data_dir or '/' in data_dir:
        return data_dir
    
    if data_dir in supported_datasets:
        return f"data/{data_dir}"
    
    return f"data/{data_dir}"


def main():
    supported_datasets = [
        'WildBETA', 'BETA', 'icews_wiki', 'icews_yago', 'YAGO-WIKI50K-1K',
        'DICEWS-200', 'en_fr', 'DBP-WIKI', 'DOREMUS', 'AGROLD'
    ]
    
    parser = argparse.ArgumentParser(
        description="HyDRA: Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Pipeline stages (with iterative refinement):
  1. Encoding and Integration
  2. Scale-Adaptive Entity Projection
  3. Multi-Scale Fusion

Stopping conditions:
  - KG1 entities < min_kg1_entities (default: 50)
  - Iterations >= max_iterations (default: 3)

Supported datasets: {', '.join(supported_datasets)}

Examples:
  python HyDRA_main.py --data_dir icews_wiki
  python HyDRA_main.py --data_dir icews_wiki --skip_s4
  python HyDRA_main.py --data_dir icews_wiki --only_s4
  python HyDRA_main.py --data_dir icews_wiki --max_iterations 5 --min_kg1_entities 100
  python HyDRA_main.py --data_dir WildBETA --multi_granularity_time --add_noise --noise_ratio 0.8
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=f"Data directory path or dataset name. Supported: {', '.join(supported_datasets)}"
    )
    
    parser.add_argument(
        "--skip_s4",
        action="store_true",
        help="Skip Stage 1: Encoding and Integration"
    )
    
    parser.add_argument(
        "--only_s4",
        action="store_true",
        help="Run only Stage 1: Encoding and Integration"
    )
    
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs for Encoding and Integration (default: 500)"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum iterations (default: 3)"
    )
    
    parser.add_argument(
        "--min_kg1_entities",
        type=int,
        default=50,
        help="Minimum KG1 entities threshold (default: 50)"
    )
    
    parser.add_argument(
        "--skip_encoding_integration",
        action="store_true",
        help="Skip encoding_and_integration step (for testing)"
    )
    
    parser.add_argument(
        "--multi_granularity_time",
        action="store_true",
        help="Enable multi-granularity temporal modeling (default: False)"
    )
    
    robustness_group = parser.add_argument_group(
        'Robustness (Sec. 5.6)',
        'Name-embedding noise injected in Stage 1 (Simple-HHEA) for embedding-degradation experiments',
    )
    robustness_group.add_argument(
        "--add_noise",
        action="store_true",
        help="Zero out a fraction of name-embedding dimensions before encoding (paper Sec. 5.6)",
    )
    robustness_group.add_argument(
        "--noise_ratio",
        type=float,
        default=0.0,
        help="Fraction of 64-d name-embedding dims to mask when --add_noise is set (0.0--1.0, e.g. 0.8 for 80%%)",
    )

    parser.add_argument(
        "--snapshot_sup_pairs",
        action="store_true",
        help="Snapshot sup_pairs after each iteration for pseudo-label tracking"
    )

    ablation_group = parser.add_argument_group('Ablation Experiments', 'Ablation options for component evaluation')
    
    ablation_group.add_argument(
        "--w/oMulti-GranularTemporalEncoder",
        dest="wo_multi_granular_temporal_encoder",
        action="store_true",
        help="Ablation: Remove multi-granular temporal encoder"
    )
    ablation_group.add_argument(
        "--w/oYearGranularity",
        dest="wo_year_granularity",
        action="store_true",
        help="Ablation: Remove year granularity"
    )
    ablation_group.add_argument(
        "--w/oDateGranularity",
        dest="wo_date_granularity",
        action="store_true",
        help="Ablation: Remove date granularity"
    )
    ablation_group.add_argument(
        "--w/oScale-AdaptiveEntityProjection",
        dest="wo_scale_adaptive_entity_projection",
        action="store_true",
        help="Ablation: Remove scale-adaptive entity projection"
    )
    ablation_group.add_argument(
        "--w/oAdaptiveTimeProjection",
        dest="wo_adaptive_time_projection",
        action="store_true",
        help="Ablation: Remove adaptive time projection"
    )
    ablation_group.add_argument(
        "--w/oAdaptiveRelationProjection",
        dest="wo_adaptive_relation_projection",
        action="store_true",
        help="Ablation: Remove adaptive relation projection"
    )
    ablation_group.add_argument(
        "--w/oMulti-ScaleHypergraphRetrieval",
        dest="wo_multi_scale_hypergraph_retrieval",
        action="store_true",
        help="Ablation: Remove multi-scale hypergraph retrieval"
    )
    ablation_group.add_argument(
        "--w/oMulti-ScaleHypergraph",
        dest="wo_multi_scale_hypergraph",
        action="store_true",
        help="Ablation: Remove multi-scale hypergraph (use single scale L1 only)"
    )
    ablation_group.add_argument(
        "--w/oMulti-ScaleInteraction-AugmentedFusion",
        dest="wo_multi_scale_interaction_augmented_fusion",
        action="store_true",
        help="Ablation: Remove multi-scale interaction-augmented fusion"
    )
    ablation_group.add_argument(
        "--w/oIntra-ScaleInteraction",
        dest="wo_intra_scale_interaction",
        action="store_true",
        help="Ablation: Remove intra-scale interaction"
    )
    ablation_group.add_argument(
        "--w/oMulti-ScaleFusionReasoning",
        dest="wo_multi_scale_fusion_reasoning",
        action="store_true",
        help="Ablation: Remove multi-scale fusion reasoning"
    )
    ablation_group.add_argument(
        "--w/oConflictDetection",
        dest="wo_conflict_detection",
        action="store_true",
        help="Ablation: Remove conflict detection"
    )
    ablation_group.add_argument(
        "--w/SingleScaleAllEqualL1",
        dest="w_single_scale_all_equal_l1",
        action="store_true",
        help="Ablation: Copy L1 candidates into L2 and L3, isolating fusion reasoning"
    )
    
    args = parser.parse_args()
    
    if args.add_noise and not (0.0 <= args.noise_ratio <= 1.0):
        parser.error("--noise_ratio must be between 0.0 and 1.0 when --add_noise is enabled")
    
    ablation_config = AblationConfig()
    ablation_applied = False
    
    if getattr(args, 'wo_multi_granular_temporal_encoder', False):
        ablation_config.apply_ablation('w/oMulti-GranularTemporalEncoder')
        ablation_applied = True
    if getattr(args, 'wo_year_granularity', False):
        ablation_config.apply_ablation('w/oYearGranularity')
        ablation_applied = True
    if getattr(args, 'wo_date_granularity', False):
        ablation_config.apply_ablation('w/oDateGranularity')
        ablation_applied = True
    if getattr(args, 'wo_scale_adaptive_entity_projection', False):
        ablation_config.apply_ablation('w/oScale-AdaptiveEntityProjection')
        ablation_applied = True
    if getattr(args, 'wo_adaptive_time_projection', False):
        ablation_config.apply_ablation('w/oAdaptiveTimeProjection')
        ablation_applied = True
    if getattr(args, 'wo_adaptive_relation_projection', False):
        ablation_config.apply_ablation('w/oAdaptiveRelationProjection')
        ablation_applied = True
    if getattr(args, 'wo_multi_scale_hypergraph_retrieval', False):
        ablation_config.apply_ablation('w/oMulti-ScaleHypergraphRetrieval')
        ablation_applied = True
    if getattr(args, 'wo_multi_scale_hypergraph', False):
        ablation_config.apply_ablation('w/oMulti-ScaleHypergraph')
        ablation_applied = True
    if getattr(args, 'wo_multi_scale_interaction_augmented_fusion', False):
        ablation_config.apply_ablation('w/oMulti-ScaleInteraction-AugmentedFusion')
        ablation_applied = True
    if getattr(args, 'wo_intra_scale_interaction', False):
        ablation_config.apply_ablation('w/oIntra-ScaleInteraction')
        ablation_applied = True
    if getattr(args, 'wo_multi_scale_fusion_reasoning', False):
        ablation_config.apply_ablation('w/oMulti-ScaleFusionReasoning')
        ablation_applied = True
    if getattr(args, 'wo_conflict_detection', False):
        ablation_config.apply_ablation('w/oConflictDetection')
        ablation_applied = True
    if getattr(args, 'w_single_scale_all_equal_l1', False):
        ablation_config.apply_ablation('w/SingleScaleAllEqualL1')
        ablation_applied = True
    
    if not ablation_applied:
        ablation_config = None
    
    if args.skip_s4 and args.only_s4:
        print("Error: --skip_s4 and --only_s4 cannot be used together")
        sys.exit(1)
    
    data_dir = normalize_data_dir(args.data_dir)
    
    if data_dir != args.data_dir:
        print(f"Dataset name '{args.data_dir}' converted to path: {data_dir}")
    
    success = run_full_pipeline(
        data_dir=data_dir,
        skip_s4=args.skip_s4,
        only_s4=args.only_s4,
        cuda=args.cuda,
        epochs=args.epochs,
        max_iterations=args.max_iterations,
        min_kg1_entities=args.min_kg1_entities,
        skip_encoding_integration=args.skip_encoding_integration,
        multi_granularity_time=args.multi_granularity_time,
        add_noise=args.add_noise,
        noise_ratio=args.noise_ratio,
        ablation_config=ablation_config,
        snapshot_sup_pairs=args.snapshot_sup_pairs
    )
    
    if success:
        print("✓ HyDRA pipeline completed successfully!")
        sys.exit(0)
    else:
        print("✗ HyDRA pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
