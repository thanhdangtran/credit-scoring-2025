"""
Modeling module for Vietnamese Credit Scoring.

This module provides tools for building credit scoring models including:
- Preprocessing: WOE transformation, feature engineering
- Model training: Logistic regression, scorecards
- Evaluation: Gini, KS, PSI metrics
"""

from .preprocessing import (
    BinningMethod,
    MissingStrategy,
    WOEBinner,
    WOETransformer,
    WOEBinStats,
    FeatureWOEResult,
    IV_THRESHOLDS,
)

__all__ = [
    # Preprocessing
    "BinningMethod",
    "MissingStrategy",
    "WOEBinner",
    "WOETransformer",
    "WOEBinStats",
    "FeatureWOEResult",
    "IV_THRESHOLDS",
]
