"""
Exchange package for interacting with cryptocurrency exchanges.

This package provides a unified interface for interacting with various
cryptocurrency exchanges, including:
- Order management
- Market data retrieval
- Account information
- WebSocket feeds
"""

import logging
from typing import Dict, Type
from .base import ExchangeClient

logger = logging.getLogger(__name__)
AVAILABLE_EXCHANGES: Dict[str, Type[ExchangeClient]] = {}

# Conditional imports for different exchanges
try:
    from .binance import BinanceClient
    AVAILABLE_EXCHANGES['binance'] = BinanceClient
except ImportError:
    logger.warning("Binance client not available. Install 'ccxt' to enable Binance support.")

try:
    from .bitget import BitgetClient
    AVAILABLE_EXCHANGES['bitget'] = BitgetClient
except ImportError:
    logger.warning("Bitget client not available. Install required dependencies to enable Bitget support.")

def get_exchange_client(exchange_name: str) -> Type[ExchangeClient]:
    """
    Get exchange client class by name.
    
    Args:
        exchange_name: Name of the exchange
        
    Returns:
        Exchange client class
        
    Raises:
        ValueError: If exchange is not supported
    """
    if exchange_name not in AVAILABLE_EXCHANGES:
        raise ValueError(
            f"Exchange '{exchange_name}' not supported. "
            f"Available exchanges: {list(AVAILABLE_EXCHANGES.keys())}"
        )
    return AVAILABLE_EXCHANGES[exchange_name]

__version__ = '1.0.0'
__all__ = ['ExchangeClient', 'get_exchange_client'] + list(AVAILABLE_EXCHANGES.keys()) 