"""Splitting a generated sequence into per-token (state, action) pairs."""

from token_attribution.extractor import TokenExtractor
from token_attribution.sa_pair_builder import SAPairBuilder
from token_attribution.token_attribution import TokenAttribution

__all__ = ["SAPairBuilder", "TokenAttribution", "TokenExtractor"]
