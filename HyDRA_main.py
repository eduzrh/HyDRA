#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDRA Main Pipeline Script

Features: Orchestrates the entire HyDRA pipeline, including:
1. S4 Training (Simple-HHEA)
2. Relation Alignment + Hypergraph Decomposition + Neural Retrieval + Multi-Scale Representation
3. Multi-Scale Fusion + Add to sup_pairs

Usage:
    python HyDRA_main.py --data_dir data/icews_wiki [options]

Examples:
    # Run complete pipeline
    python HyDRA_main.py --data_dir data/icews_wiki

    # Skip S4 training (if results already exist)
    python HyDRA_main.py --data_dir data/icews_wiki --skip_s4

    # Only run S4 training
    python HyDRA_main.py --data_dir data/icews_wiki --only_s4

    # Only run retrieval and fusion (skip S4)
    python HyDRA_main.py --data_dir data/icews_wiki --skip_s4
"""

import os
import sys
import argparse
import subprocess


def check_s4_output(data_dir):
    """
    Check if S4 output file exists
    
    Args:
        data_dir: Data directory path
        
    Returns:
        bool: Whether S4 output file exists
    """
    s4_output_file = os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')
    return os.path.exists(s4_output_file) and os.path.getsize(s4_output_file) > 0


def count_unique_kg1_entities(data_dir):
    """
    Count unique KG1 entities in integration_top_pair.txt
    
    Args:
        data_dir: Data directory path
        
    Returns:
        int: Number of unique KG1 entities, returns 0 if file does not exist
    """
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
        print(f"Warning: Error reading S4 output file: {e}")
        return 0
    
    return len(kg1_entities)


def ensure_ent_ids_1_restored(data_dir):
    """
    Ensure ent_ids_1 file is restored (if it was temporarily replaced)
    
    Args:
        data_dir: Data directory path
        
    Returns:
        bool: Whether the file is in correct state
    """
    ent_ids_1_path = os.path.join(data_dir, "ent_ids_1")
    backup_path = os.path.join(data_dir, "ent_ids_1.backup")
    
    # If backup file exists, ent_ids_1 may have been replaced and needs to be restored
    if os.path.exists(backup_path):
        import shutil
        print(f"  Warning: Found backup file, restoring ent_ids_1 from backup...")
        shutil.move(backup_path, ent_ids_1_path)
        print(f"  Restored ent_ids_1 from backup")
    
    # Check if ent_ids_1 file is normal (should have reasonable number of lines, e.g., >1000)
    if os.path.exists(ent_ids_1_path):
        with open(ent_ids_1_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
            if line_count < 1000:
                print(f"  Warning: ent_ids_1 has only {line_count} lines, which seems too small")
                print(f"  This might cause IndexError in S4 training")
                return False
        return True
    else:
        print(f"  Error: ent_ids_1 file not found: {ent_ids_1_path}")
        return False


def run_s4_training(data_dir, cuda=0, epochs=500):
    """
    Run S4 training (Simple-HHEA)
    
    Args:
        data_dir: Data directory path
        cuda: CUDA device ID
        epochs: Number of training epochs
        
    Returns:
        bool: Whether successful
    """
    print("\n" + "=" * 80)
    print("Step 1: Running S4 Training (Simple-HHEA)")
    print("=" * 80 + "\n")
    
    # Ensure ent_ids_1 file is correctly restored (if it was temporarily replaced)
    print("Checking ent_ids_1 file before S4 training...")
    if not ensure_ent_ids_1_restored(data_dir):
        print("  Error: ent_ids_1 file is not in correct state. Please check the file manually.")
        return False
    
    # Check if run_s4_standalone.py exists
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
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error running S4 training: {e}")
            return False
    else:
        print(f"Warning: {s4_script} not found. Please run S4 training manually.")
        print(f"Expected output: {os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')}")
        return False


def run_s4_to_retrieval(data_dir, iteration=1):
    """
    Run S4 to Retrieval pipeline (Relation Alignment + Hypergraph Decomposition + Retrieval + Multi-Scale Representation)
    
    Args:
        data_dir: Data directory path
        iteration: Current iteration number (default: 1)
        
    Returns:
        bool: Whether successful
    """
    print("\n" + "=" * 80)
    print("Step 2: Running S4 to Retrieval (Relation Alignment + Hypergraph Decomposition + Retrieval + Multi-Scale Representation)")
    print("=" * 80 + "\n")
    
    try:
        from scale_adaptive_entity_projection.entity_projection import s4_to_retrieval
        success = s4_to_retrieval(data_dir, iteration=iteration)
        return success
    except ImportError as e:
        print(f"Error importing s4_to_retrieval: {e}")
        return False
    except Exception as e:
        print(f"Error running s4_to_retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_multi_scale_fusion(data_dir):
    """
    Run multi-scale fusion (including automatic addition to sup_pairs)
    
    Args:
        data_dir: Data directory path
        
    Returns:
        bool: Whether successful
    """
    print("\n" + "=" * 80)
    print("Step 3: Running Multi-Scale Fusion (with automatic sup_pairs update)")
    print("=" * 80 + "\n")
    
    try:
        from multi_scale_fusion.multi_scale_fusion import multi_scale_fusion
        aligned_pairs = multi_scale_fusion(data_dir)
        return aligned_pairs is not None and len(aligned_pairs) > 0
    except ImportError as e:
        print(f"Error importing multi_scale_fusion: {e}")
        return False
    except Exception as e:
        print(f"Error running multi_scale_fusion: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_pipeline(data_dir, skip_s4=False, only_s4=False, cuda=0, epochs=500, max_iterations=3, min_kg1_entities=50):
    """
    Run complete HyDRA pipeline (supports iterative loops)
    
    Args:
        data_dir: Data directory path
        skip_s4: Whether to skip S4 training
        only_s4: Whether to only run S4 training
        cuda: CUDA device ID (for S4 training)
        epochs: Number of training epochs (for S4 training)
        max_iterations: Maximum number of iterations (default: 3)
        min_kg1_entities: Minimum KG1 entity count threshold (default: 50)
        
    Returns:
        bool: Whether successful
    """
    print("\n" + "=" * 80)
    print("HyDRA: Complete Pipeline (with iteration control)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Skip S4: {skip_s4}")
    print(f"Only S4: {only_s4}")
    print(f"Max iterations: {max_iterations}")
    print(f"Min KG1 entities threshold: {min_kg1_entities}")
    print("=" * 80 + "\n")
    
    # Check data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return False
    
    # If only running S4, do not loop
    if only_s4:
        success_steps = []
        
        if not skip_s4:
            if check_s4_output(data_dir):
                print("✓ S4 output file already exists, skipping S4 training")
                print(f"  File: {os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')}")
                success_steps.append("S4 (already exists)")
            else:
                if run_s4_training(data_dir, cuda=cuda, epochs=epochs):
                    success_steps.append("S4 Training")
                else:
                    print("✗ S4 training failed")
                    return False
        else:
            if not check_s4_output(data_dir):
                print("✗ Error: S4 output file not found and --skip_s4 is set")
                return False
            else:
                print("✓ S4 output file exists, skipping S4 training")
                success_steps.append("S4 (skipped)")
        
        print("\n" + "=" * 80)
        print("✓ Only S4 training completed")
        print("=" * 80 + "\n")
        return True
    
    # Iterative loop
    iteration = 0
    all_success_steps = []
    
    while iteration < max_iterations:
        iteration += 1
        print("\n" + "=" * 80)
        print(f"Iteration {iteration}/{max_iterations}")
        print("=" * 80 + "\n")
        
        success_steps = []
        
        # Step 1: S4 Training
        if not skip_s4:
            if check_s4_output(data_dir) and iteration > 1:
                print("✓ S4 output file already exists from previous iteration")
                print(f"  File: {os.path.join(data_dir, 'message_pool', 'integration_top_pair.txt')}")
                success_steps.append("S4 (already exists)")
            else:
                if run_s4_training(data_dir, cuda=cuda, epochs=epochs):
                    success_steps.append("S4 Training")
                else:
                    print("✗ S4 training failed")
                    if iteration == 1:
                        return False
                    # If not the first iteration, continue with existing results
                    print("  Continuing with existing files...")
        else:
            if not check_s4_output(data_dir):
                print("✗ Error: S4 output file not found and --skip_s4 is set")
                if iteration == 1:
                    return False
                print("  Continuing with existing files...")
            else:
                print("✓ S4 output file exists, skipping S4 training")
                success_steps.append("S4 (skipped)")
        
        # Check KG1 entity count in S4 output (before preparing next step)
        kg1_count = count_unique_kg1_entities(data_dir)
        print(f"\nUnique KG1 entities in S4 output: {kg1_count}")
        
        # Stopping condition 1: KG1 entity count is less than threshold
        if kg1_count < min_kg1_entities:
            print(f"\n{'=' * 80}")
            print(f"Stopping condition met: KG1 entities ({kg1_count}) < threshold ({min_kg1_entities})")
            print(f"{'=' * 80}\n")
            all_success_steps.extend(success_steps)
            break
        
        # Step 2: S4 to Retrieval
        if run_s4_to_retrieval(data_dir, iteration=iteration):
            success_steps.append("S4 to Retrieval")
        else:
            print("✗ S4 to Retrieval failed")
            if iteration == 1:
                return False
            # Continue to next iteration
            continue
        
        # Step 3: Multi-Scale Fusion
        if run_multi_scale_fusion(data_dir):
            success_steps.append("Multi-Scale Fusion")
        else:
            print("✗ Multi-Scale Fusion failed")
            if iteration == 1:
                return False
            # Continue to next iteration
            continue
        
        all_success_steps.extend(success_steps)
        
        # Check if maximum iterations reached
        if iteration >= max_iterations:
            print(f"\n{'=' * 80}")
            print(f"Reached maximum iterations: {max_iterations}")
            print(f"{'=' * 80}\n")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("HyDRA Pipeline Summary")
    print("=" * 80)
    print(f"Total iterations: {iteration}/{max_iterations}")
    print(f"Final KG1 entities count: {count_unique_kg1_entities(data_dir)}")
    print(f"Completed steps: {', '.join(all_success_steps)}")
    print("=" * 80 + "\n")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="HyDRA: Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete pipeline includes (supports iterative loops):
  1. S4 Training (Simple-HHEA)
  2. Relation Alignment + Hypergraph Decomposition + Neural Retrieval + Multi-Scale Representation
  3. Multi-Scale Fusion + Automatic addition to sup_pairs

Stopping conditions:
  - If unique KG1 entity count in S4 output < min_kg1_entities (default 50), stop loop
  - If iteration count >= max_iterations (default 3), stop loop

Example usage:
  # Run complete pipeline (max 3 iterations)
  python HyDRA_main.py --data_dir data/icews_wiki

  # Skip S4 training (if results already exist)
  python HyDRA_main.py --data_dir data/icews_wiki --skip_s4

  # Only run S4 training
  python HyDRA_main.py --data_dir data/icews_wiki --only_s4

  # Specify maximum iterations and minimum entity count threshold
  python HyDRA_main.py --data_dir data/icews_wiki --max_iterations 5 --min_kg1_entities 100

  # Specify CUDA device and training epochs
  python HyDRA_main.py --data_dir data/icews_wiki --cuda 0 --epochs 500
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory path (e.g., data/icews_wiki)"
    )
    
    parser.add_argument(
        "--skip_s4",
        action="store_true",
        help="Skip S4 training step (if integration_top_pair.txt already exists)"
    )
    
    parser.add_argument(
        "--only_s4",
        action="store_true",
        help="Only run S4 training step"
    )
    
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device ID (for S4 training, default: 0)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="S4 training epochs (default: 500)"
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum number of iterations (default: 3)"
    )
    
    parser.add_argument(
        "--min_kg1_entities",
        type=int,
        default=50,
        help="Minimum KG1 entity count threshold (default: 50, loop stops if below this value)"
    )
    
    args = parser.parse_args()
    
    # Check for parameter conflicts
    if args.skip_s4 and args.only_s4:
        print("Error: --skip_s4 and --only_s4 cannot be used together")
        sys.exit(1)
    
    # Run complete pipeline
    success = run_full_pipeline(
        data_dir=args.data_dir,
        skip_s4=args.skip_s4,
        only_s4=args.only_s4,
        cuda=args.cuda,
        epochs=args.epochs,
        max_iterations=args.max_iterations,
        min_kg1_entities=args.min_kg1_entities
    )
    
    if success:
        print("✓ HyDRA pipeline completed successfully!")
        sys.exit(0)
    else:
        print("✗ HyDRA pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

