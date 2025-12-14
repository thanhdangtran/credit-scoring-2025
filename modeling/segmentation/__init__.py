"""
Segmentation module for credit scoring.

This module provides customer segmentation tools for credit risk modeling:
- CHAID (Chi-square Automatic Interaction Detection)
- CART (Classification and Regression Trees)

Segmentation is used to:
- Identify distinct customer groups with different risk profiles
- Build segment-specific scorecards
- Improve model performance through population stratification
"""

from .chaid_segmenter import (
    # Dataclasses
    CHAIDNode,
    SegmentProfile,
    # Classes
    CHAIDSegmenter,
    VietnameseCreditSegmenter,
    # Constants
    DEFAULT_ALPHA,
    VIETNAMESE_SEGMENT_NAMES,
)

from .cart_segmenter import (
    # Enums
    SplitCriterion,
    PruningStrategy,
    # Dataclasses
    CARTNode,
    SegmentStats,
    # Classes
    CARTSegmenter,
    CustomCARTSegmenter,
)

__all__ = [
    # CHAID Dataclasses
    "CHAIDNode",
    "SegmentProfile",
    # CHAID Classes
    "CHAIDSegmenter",
    "VietnameseCreditSegmenter",
    # CHAID Constants
    "DEFAULT_ALPHA",
    "VIETNAMESE_SEGMENT_NAMES",
    # CART Enums
    "SplitCriterion",
    "PruningStrategy",
    # CART Dataclasses
    "CARTNode",
    "SegmentStats",
    # CART Classes
    "CARTSegmenter",
    "CustomCARTSegmenter",
]
