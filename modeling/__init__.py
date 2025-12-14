"""
Modeling module for Vietnamese Credit Scoring.

This module provides tools for building credit scoring models including:
- Preprocessing: WOE transformation, feature engineering
- Segmentation: CHAID-based customer segmentation
- Model training: Logistic regression, scorecards
- Evaluation: Gini, KS, PSI metrics
"""

from .preprocessing import (
    # WOE Transformer
    BinningMethod,
    MissingStrategy,
    WOEBinner,
    WOETransformer,
    WOEBinStats,
    FeatureWOEResult,
    IV_THRESHOLDS,
    # Feature Engineering
    TimeSeriesFeatureEngineer,
    StaticFeatureEngineer,
    MissingFeatureEngineer,
    CreditFeatureEngineer,
)

from .segmentation import (
    # CHAID
    CHAIDNode,
    SegmentProfile,
    CHAIDSegmenter,
    VietnameseCreditSegmenter,
    # CART
    CARTNode,
    SegmentStats,
    CARTSegmenter,
    CustomCARTSegmenter,
)

from .scoring import (
    # Classes
    MulticollinearityChecker,
    StepwiseSelector,
    CreditLogisticModel,
    # Dataclasses
    ModelCoefficient,
    ModelSummary,
)

__all__ = [
    # Preprocessing - WOE
    "BinningMethod",
    "MissingStrategy",
    "WOEBinner",
    "WOETransformer",
    "WOEBinStats",
    "FeatureWOEResult",
    "IV_THRESHOLDS",
    # Preprocessing - Feature Engineering
    "TimeSeriesFeatureEngineer",
    "StaticFeatureEngineer",
    "MissingFeatureEngineer",
    "CreditFeatureEngineer",
    # Segmentation - CHAID
    "CHAIDNode",
    "SegmentProfile",
    "CHAIDSegmenter",
    "VietnameseCreditSegmenter",
    # Segmentation - CART
    "CARTNode",
    "SegmentStats",
    "CARTSegmenter",
    "CustomCARTSegmenter",
    # Scoring
    "MulticollinearityChecker",
    "StepwiseSelector",
    "CreditLogisticModel",
    "ModelCoefficient",
    "ModelSummary",
]
