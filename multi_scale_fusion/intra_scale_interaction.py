#!/usr/bin/env python3
"""
Intra-Scale Interaction and Conflict Detection Module

Features:
1. Intra-scale interaction: Analyze interactions and consistency between candidate entities within each scale
2. Conflict detection: Detect cross-scale conflicts and intra-scale conflicts
3. Generate conflict reports to guide multi-scale fusion decisions
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import os
import sys

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def analyze_intra_scale_interaction(candidate_info: Dict[str, List[int]], 
                                    ent_names_2: Dict[int, str]) -> Dict[str, Dict]:
    """
    Analyze intra-scale interaction
    
    For each scale, analyze:
    1. Ranking distribution of candidate entities
    2. Correlation between candidate entities (if duplicate entities appear in multiple scales)
    3. Confidence distribution within the scale
    
    Args:
        candidate_info: {'L1': [kg2_ids], 'L2': [kg2_ids], 'L3': [kg2_ids]}
        ent_names_2: KG2 entity name dictionary
        
    Returns:
        dict: Interaction analysis results for each scale
        {
            'L1': {
                'candidate_count': int,
                'candidates': [kg2_ids],
                'top_candidate': kg2_id or None,
                'confidence_signal': str  # 'strong', 'medium', 'weak'
            },
            ...
        }
    """
    scale_analysis = {}
    
    for scale in ['L1', 'L2', 'L3']:
        candidates = candidate_info.get(scale, [])
        
        analysis = {
            'candidate_count': len(candidates),
            'candidates': candidates,
            'top_candidate': candidates[0] if candidates else None,
            'confidence_signal': 'weak'
        }
        
        # Determine confidence signal based on candidate count
        if len(candidates) == 1:
            # Single candidate, high confidence
            analysis['confidence_signal'] = 'strong'
        elif len(candidates) <= 3:
            # Few candidates, medium confidence
            analysis['confidence_signal'] = 'medium'
        else:
            # Multiple candidates, need further judgment
            analysis['confidence_signal'] = 'weak'
        
        scale_analysis[scale] = analysis
    
    return scale_analysis


def detect_cross_scale_conflicts(candidate_info: Dict[str, List[int]],
                                 scale_analysis: Dict[str, Dict]) -> Dict:
    """
    Detect cross-scale conflicts
    
    Detect whether the TOP-1 candidate entities recommended by different scales are consistent
    
    Args:
        candidate_info: {'L1': [kg2_ids], 'L2': [kg2_ids], 'L3': [kg2_ids]}
        scale_analysis: Intra-scale interaction analysis results
        
    Returns:
        dict: Conflict detection results
        {
            'has_conflict': bool,
            'conflict_type': str,  # 'cross_scale', 'intra_scale', 'none'
            'conflicting_scales': List[str],
            'top_candidates': {scale: kg2_id},
            'consensus_candidate': kg2_id or None,
            'conflict_details': str
        }
    """
    # Get TOP-1 candidate from each scale
    top_candidates = {}
    for scale in ['L1', 'L2', 'L3']:
        candidates = candidate_info.get(scale, [])
        if candidates:
            top_candidates[scale] = candidates[0]
    
    if len(top_candidates) <= 1:
        # Only 0 or 1 scale has candidates, no conflict
        return {
            'has_conflict': False,
            'conflict_type': 'none',
            'conflicting_scales': [],
            'top_candidates': top_candidates,
            'consensus_candidate': list(top_candidates.values())[0] if top_candidates else None,
            'conflict_details': 'No conflict: insufficient scales for comparison'
        }
    
    # Check if TOP-1 candidates are consistent
    top_values = list(top_candidates.values())
    unique_tops = set(top_values)
    
    if len(unique_tops) == 1:
        # All scales agree on TOP-1, no conflict
        return {
            'has_conflict': False,
            'conflict_type': 'none',
            'conflicting_scales': [],
            'top_candidates': top_candidates,
            'consensus_candidate': top_values[0],
            'conflict_details': 'No conflict: all scales agree on top candidate'
        }
    
    # Detect conflicts
    # Find inconsistent scales
    conflicting_scales = []
    consensus_candidate = None
    
    # Count occurrences of each candidate
    candidate_counts = defaultdict(int)
    for kg2_id in top_values:
        candidate_counts[kg2_id] += 1
    
    # Find candidate with most occurrences (majority voting)
    if candidate_counts:
        consensus_candidate = max(candidate_counts.items(), key=lambda x: x[1])[0]
    
    # Find scales inconsistent with consensus
    for scale, kg2_id in top_candidates.items():
        if kg2_id != consensus_candidate:
            conflicting_scales.append(scale)
    
    conflict_details = f"Conflict detected: {len(conflicting_scales)} scale(s) disagree. "
    conflict_details += f"Consensus candidate: {consensus_candidate}, "
    conflict_details += f"Conflicting scales: {conflicting_scales}"
    
    return {
        'has_conflict': True,
        'conflict_type': 'cross_scale',
        'conflicting_scales': conflicting_scales,
        'top_candidates': top_candidates,
        'consensus_candidate': consensus_candidate,
        'conflict_details': conflict_details
    }


def detect_intra_scale_conflicts(candidate_info: Dict[str, List[int]],
                                 scale_analysis: Dict[str, Dict]) -> Dict:
    """
    Detect intra-scale conflicts
    
    Detect whether there are multiple high-confidence but contradictory candidate entities within a single scale
    (e.g., many candidate entities, and the top few candidates all have high confidence)
    
    Args:
        candidate_info: {'L1': [kg2_ids], 'L2': [kg2_ids], 'L3': [kg2_ids]}
        scale_analysis: Intra-scale interaction analysis results
        
    Returns:
        dict: Intra-scale conflict detection results
        {
            'conflicted_scales': List[str],
            'conflict_details': {scale: str}
        }
    """
    conflicted_scales = []
    conflict_details = {}
    
    for scale in ['L1', 'L2', 'L3']:
        candidates = candidate_info.get(scale, [])
        
        # If candidate count is high (>3), intra-scale conflict may exist
        # Or if candidate count is moderate (2-3), but needs further judgment
        if len(candidates) > 3:
            conflicted_scales.append(scale)
            conflict_details[scale] = f"Multiple candidates ({len(candidates)}) in {scale} scale, may need further verification"
    
    return {
        'conflicted_scales': conflicted_scales,
        'conflict_details': conflict_details
    }


def generate_conflict_summary(cross_scale_conflicts: Dict,
                              intra_scale_conflicts: Dict,
                              scale_analysis: Dict[str, Dict]) -> str:
    """
    Generate conflict summary for LLM Prompt
    
    Args:
        cross_scale_conflicts: Cross-scale conflict detection results
        intra_scale_conflicts: Intra-scale conflict detection results
        scale_analysis: Intra-scale interaction analysis results
        
    Returns:
        str: Conflict summary text
    """
    summary_parts = []
    
    # Cross-scale conflict information
    if cross_scale_conflicts['has_conflict']:
        summary_parts.append("⚠️ Cross-Scale Conflict Detected:")
        summary_parts.append(f"  - {cross_scale_conflicts['conflict_details']}")
        summary_parts.append(f"  - Consensus candidate: {cross_scale_conflicts['consensus_candidate']}")
        summary_parts.append(f"  - Please carefully evaluate consistency across scales")
    else:
        summary_parts.append("✓ No cross-scale conflicts: All scales agree on top candidate")
    
    # Intra-scale conflict information
    if intra_scale_conflicts['conflicted_scales']:
        summary_parts.append("\n⚠️ Intra-Scale Conflicts Detected:")
        for scale in intra_scale_conflicts['conflicted_scales']:
            summary_parts.append(f"  - {intra_scale_conflicts['conflict_details'][scale]}")
    
    # Scale analysis summary
    summary_parts.append("\nScale Analysis Summary:")
    for scale in ['L1', 'L2', 'L3']:
        analysis = scale_analysis.get(scale, {})
        if analysis.get('candidate_count', 0) > 0:
            signal = analysis.get('confidence_signal', 'unknown')
            count = analysis.get('candidate_count', 0)
            top = analysis.get('top_candidate', 'N/A')
            summary_parts.append(f"  - {scale} scale: {count} candidate(s), confidence: {signal}, top: {top}")
    
    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Test code
    test_candidate_info = {
        'L1': [25030],
        'L2': [24118, 23854, 25293, 26471, 25030],
        'L3': [24118, 23854, 25293, 26471, 25030]
    }
    
    test_ent_names_2 = {25030: "Gina Raimondo", 24118: "Vietnamese people", 23854: "Taufa'ahau Tupou IV"}
    
    scale_analysis = analyze_intra_scale_interaction(test_candidate_info, test_ent_names_2)
    print("Scale Analysis:", scale_analysis)
    
    cross_conflicts = detect_cross_scale_conflicts(test_candidate_info, scale_analysis)
    print("\nCross-Scale Conflicts:", cross_conflicts)
    
    intra_conflicts = detect_intra_scale_conflicts(test_candidate_info, scale_analysis)
    print("\nIntra-Scale Conflicts:", intra_conflicts)
    
    summary = generate_conflict_summary(cross_conflicts, intra_conflicts, scale_analysis)
    print("\nConflict Summary:")
    print(summary)


