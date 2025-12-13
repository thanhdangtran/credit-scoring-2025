"""
Preprocessing module for credit scoring model development.

This module provides tools for feature transformation including:
- WOE (Weight of Evidence) encoding
- Binning strategies
- Missing value handling
"""

from .woe_transformer import (
    # Enums
    BinningMethod,
    MissingStrategy,
    # Constants
    IV_THRESHOLDS,
    SMOOTHING_CONSTANT,
    # Dataclasses
    WOEBinStats,
    FeatureWOEResult,
    # Classes
    WOEBinner,
    WOETransformer,
)

__all__ = [
    # Enums
    "BinningMethod",
    "MissingStrategy",
    # Constants
    "IV_THRESHOLDS",
    "SMOOTHING_CONSTANT",
    # Dataclasses
    "WOEBinStats",
    "FeatureWOEResult",
    # Classes
    "WOEBinner",
    "WOETransformer",
]
