"""
Ablation experiment configuration module.
"""

class AblationConfig:
    """Ablation experiment configuration class."""
    
    def __init__(self):
        self.use_multi_granular_temporal_encoder = True
        self.use_year_granularity = True
        self.use_date_granularity = True
        
        self.use_scale_adaptive_entity_projection = True
        self.use_adaptive_time_projection = True
        self.use_adaptive_relation_projection = True
        
        self.use_multi_scale_hypergraph_retrieval = True
        self.use_multi_scale_hypergraph = True
        
        self.use_multi_scale_interaction_augmented_fusion = True
        self.use_intra_scale_interaction = True
        self.use_multi_scale_fusion_reasoning = True
        self.use_conflict_detection = True
    
    def apply_ablation(self, ablation_name):
        """Apply ablation configuration."""
        if ablation_name == 'w/oMulti-GranularTemporalEncoder':
            self.use_multi_granular_temporal_encoder = False
            self.use_year_granularity = False
            self.use_date_granularity = False
        elif ablation_name == 'w/oYearGranularity':
            self.use_year_granularity = False
        elif ablation_name == 'w/oDateGranularity':
            self.use_date_granularity = False
        elif ablation_name == 'w/oScale-AdaptiveEntityProjection':
            self.use_scale_adaptive_entity_projection = False
            self.use_adaptive_time_projection = False
            self.use_adaptive_relation_projection = False
        elif ablation_name == 'w/oAdaptiveTimeProjection':
            self.use_adaptive_time_projection = False
        elif ablation_name == 'w/oAdaptiveRelationProjection':
            self.use_adaptive_relation_projection = False
        elif ablation_name == 'w/oMulti-ScaleHypergraphRetrieval':
            self.use_multi_scale_hypergraph_retrieval = False
        elif ablation_name == 'w/oMulti-ScaleHypergraph':
            self.use_multi_scale_hypergraph = False
        elif ablation_name == 'w/oMulti-ScaleInteraction-AugmentedFusion':
            self.use_multi_scale_interaction_augmented_fusion = False
            self.use_intra_scale_interaction = False
            self.use_multi_scale_fusion_reasoning = False
            self.use_conflict_detection = False
        elif ablation_name == 'w/oIntra-ScaleInteraction':
            self.use_intra_scale_interaction = False
        elif ablation_name == 'w/oMulti-ScaleFusionReasoning':
            self.use_multi_scale_fusion_reasoning = False
        elif ablation_name == 'w/oConflictDetection':
            self.use_conflict_detection = False
        else:
            raise ValueError(f"Unknown ablation name: {ablation_name}")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'use_multi_granular_temporal_encoder': self.use_multi_granular_temporal_encoder,
            'use_year_granularity': self.use_year_granularity,
            'use_date_granularity': self.use_date_granularity,
            'use_scale_adaptive_entity_projection': self.use_scale_adaptive_entity_projection,
            'use_adaptive_time_projection': self.use_adaptive_time_projection,
            'use_adaptive_relation_projection': self.use_adaptive_relation_projection,
            'use_multi_scale_hypergraph_retrieval': self.use_multi_scale_hypergraph_retrieval,
            'use_multi_scale_hypergraph': self.use_multi_scale_hypergraph,
            'use_multi_scale_interaction_augmented_fusion': self.use_multi_scale_interaction_augmented_fusion,
            'use_intra_scale_interaction': self.use_intra_scale_interaction,
            'use_multi_scale_fusion_reasoning': self.use_multi_scale_fusion_reasoning,
            'use_conflict_detection': self.use_conflict_detection,
        }
    
    def get_description(self):
        """Get description of current configuration."""
        parts = []
        if not self.use_multi_granular_temporal_encoder:
            parts.append("w/o Multi-Granular Temporal Encoder")
        if not self.use_year_granularity:
            parts.append("w/o Year Granularity")
        if not self.use_date_granularity:
            parts.append("w/o Date Granularity")
        if not self.use_scale_adaptive_entity_projection:
            parts.append("w/o Scale-Adaptive Entity Projection")
        if not self.use_adaptive_time_projection:
            parts.append("w/o Adaptive Time Projection")
        if not self.use_adaptive_relation_projection:
            parts.append("w/o Adaptive Relation Projection")
        if not self.use_multi_scale_hypergraph_retrieval:
            parts.append("w/o Multi-Scale Hypergraph Retrieval")
        if not self.use_multi_scale_hypergraph:
            parts.append("w/o Multi-Scale Hypergraph")
        if not self.use_multi_scale_interaction_augmented_fusion:
            parts.append("w/o Multi-Scale Interaction-Augmented Fusion")
        if not self.use_intra_scale_interaction:
            parts.append("w/o Intra-Scale Interaction")
        if not self.use_multi_scale_fusion_reasoning:
            parts.append("w/o Multi-Scale Fusion Reasoning")
        if not self.use_conflict_detection:
            parts.append("w/o Conflict Detection")
        
        if parts:
            return "Ablation: " + ", ".join(parts)
        return "Full Model (no ablation)"
