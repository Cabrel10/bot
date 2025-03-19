"""Strategies module containing various trading strategy implementations."""

from .base_strategy import BaseStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy

__all__ = [
    'BaseStrategy',
    'MeanReversionStrategy',
    'TrendFollowingStrategy'
]