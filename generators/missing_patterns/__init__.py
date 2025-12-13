"""
Missing data pattern generators for Vietnamese Credit Scoring Synthetic Data.

This module provides generators for creating realistic missing data patterns
following MCAR, MAR, and MNAR mechanisms specific to Vietnamese credit markets.
"""

from .mnar_generator import (
    # Enums
    MissingMechanism,
    MissingCategory,
    # Dataclasses
    MNARRule,
    MissingReport,
    # Generator
    MNARPatternGenerator,
    # Predefined rules
    VIETNAMESE_CREDIT_MNAR_RULES,
    TELECOM_MNAR_RULES,
    THIN_FILE_MNAR_RULES,
    ALL_MNAR_RULES,
)

__all__ = [
    # Enums
    "MissingMechanism",
    "MissingCategory",
    # Dataclasses
    "MNARRule",
    "MissingReport",
    # Generator
    "MNARPatternGenerator",
    # Predefined rules
    "VIETNAMESE_CREDIT_MNAR_RULES",
    "TELECOM_MNAR_RULES",
    "THIN_FILE_MNAR_RULES",
    "ALL_MNAR_RULES",
]
