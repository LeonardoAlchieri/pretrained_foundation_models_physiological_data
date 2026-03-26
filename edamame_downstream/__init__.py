"""
edamame_downstream: Pretrained Foundation Models for Physiological Data Analysis
================================================================================

This package provides tools for analyzing physiological signals (especially EDA)
using pretrained foundation models including MOMENT, Chronos, PatchTSMixer, and Mantis.

Key modules:
- data: Dataset loading and preprocessing
- feature_extraction: Foundation model feature extractors
- engine: Training and evaluation framework
- validation: Cross-validation strategies
- model: Machine learning models and classifiers
- preprocessing: Signal preprocessing utilities
- utils: Utility functions and configurations
"""

__version__ = "0.1.0"
__author__ = "Leonardo Alchieri"
__email__ = "leonardo.alchieri@usi.ch"

from edamame_downstream.data import EDAMAMEDataset
from edamame_downstream.engine import Engine

__all__ = [
    "EDAMAMEDataset",
    "Engine",
    "__version__",
]
