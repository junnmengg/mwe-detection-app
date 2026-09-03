"""Deprecated compatibility shim.

The contents of this module moved to :mod:`inference` (model loading and
prediction) and :mod:`text_processing` (pure tag post-processing) in v1.1.
This file only forwards the old names and will be removed in a future release.
"""

from __future__ import annotations

import warnings

from inference import LoadedModel, load_model, predict
from text_processing import (
    decode_predictions,
    extract_mwes,
    is_punctuation,
)

warnings.warn(
    "utils.py is deprecated; import from 'inference' or 'text_processing' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LoadedModel",
    "decode_predictions",
    "extract_mwes",
    "is_punctuation",
    "load_model",
    "predict",
]
