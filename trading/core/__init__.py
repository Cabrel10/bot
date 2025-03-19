"""
Core module for trading operations.
"""

from pathlib import Path
from typing import Dict, Optional

from .data_types import (
    MarketData,
    OHLCV,
    OrderData,
    TradeData,
    PositionData,
    StrategyParameters,
    BacktestResult
)

from .types import (
    ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT,
    ORDER_SIDE_BUY, ORDER_SIDE_SELL,
    TIMEFRAME_1M, TIMEFRAME_5M, TIMEFRAME_15M, TIMEFRAME_30M,
    TIMEFRAME_1H, TIMEFRAME_4H, TIMEFRAME_1D, TIMEFRAME_1W, TIMEFRAME_1MONTH
)

from .exchanges import BaseExchange
from .strategies import BaseStrategy
from .risk import RiskManager
from .position import PositionManager

ROOT_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"

__all__ = [
    'MarketData',
    'OHLCV',
    'OrderData',
    'TradeData',
    'PositionData',
    'StrategyParameters',
    'BacktestResult',
    'BaseExchange',
    'BaseStrategy',
    'RiskManager',
    'PositionManager',
    'ORDER_TYPE_MARKET', 'ORDER_TYPE_LIMIT', 'ORDER_TYPE_STOP', 'ORDER_TYPE_STOP_LIMIT',
    'ORDER_SIDE_BUY', 'ORDER_SIDE_SELL',
    'TIMEFRAME_1M', 'TIMEFRAME_5M', 'TIMEFRAME_15M', 'TIMEFRAME_30M',
    'TIMEFRAME_1H', 'TIMEFRAME_4H', 'TIMEFRAME_1D', 'TIMEFRAME_1W', 'TIMEFRAME_1MONTH'
]