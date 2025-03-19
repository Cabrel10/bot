"""
Package de gestion des données pour le trading.
"""

from .exchange_config import (
    SUPPORTED_EXCHANGES,
    REQUIRED_FEATURES,
    TECHNICAL_FEATURES,
    DEFAULT_TIMEFRAME,
    DEFAULT_EXCHANGE,
    DEFAULT_PAIRS
)
from .historical_data import HistoricalDataManager
from .dataset_manager import DatasetManager
from .web_interface import DataCollectionUI

__all__ = [
    'SUPPORTED_EXCHANGES',
    'REQUIRED_FEATURES',
    'TECHNICAL_FEATURES',
    'DEFAULT_TIMEFRAME',
    'DEFAULT_EXCHANGE',
    'DEFAULT_PAIRS',
    'HistoricalDataManager',
    'DatasetManager',
    'DataCollectionUI'
]