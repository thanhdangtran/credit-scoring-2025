"""
Time series generators for Vietnamese Credit Scoring Synthetic Data.

This module provides generators for creating realistic time series data
including banking transactions, telecom usage, and payment history.
"""

from .transaction_series import (
    TransactionSeriesGenerator,
    VietnameseCalendar,
    SeasonalityType,
    TrendType,
    OutputFormat,
    VIETNAMESE_HOLIDAYS,
    TET_MONTHS,
)

from .behavioral_series import (
    BehavioralSeriesGenerator,
    BehaviorPattern,
    DPDCategory,
    TrendDirection,
    CustomerBehaviorProfile,
    DPD_RANGES,
    PATTERN_DEFAULT_PROB,
)

__all__ = [
    # Transaction series
    "TransactionSeriesGenerator",
    "VietnameseCalendar",
    "SeasonalityType",
    "TrendType",
    "OutputFormat",
    "VIETNAMESE_HOLIDAYS",
    "TET_MONTHS",
    # Behavioral series
    "BehavioralSeriesGenerator",
    "BehaviorPattern",
    "DPDCategory",
    "TrendDirection",
    "CustomerBehaviorProfile",
    "DPD_RANGES",
    "PATTERN_DEFAULT_PROB",
]
