"""
Trading Bot Package
"""

from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Version info
__version__ = "1.0.0"

# Import main components
from .core import (
    MarketData,
    OHLCV,
    OrderData,
    TradeData,
    PositionData,
    StrategyParameters,
    BacktestResult,
    ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, ORDER_TYPE_STOP, ORDER_TYPE_STOP_LIMIT,
    ORDER_SIDE_BUY, ORDER_SIDE_SELL,
    TIMEFRAME_1M, TIMEFRAME_5M, TIMEFRAME_15M, TIMEFRAME_30M,
    TIMEFRAME_1H, TIMEFRAME_4H, TIMEFRAME_1D, TIMEFRAME_1W, TIMEFRAME_1MONTH
)

from .models import HybridModel
from .services import BacktestService, DataService, ExecutionService

__all__ = [
    'MarketData',
    'OHLCV',
    'OrderData',
    'TradeData',
    'PositionData',
    'StrategyParameters',
    'BacktestResult',
    'ORDER_TYPE_MARKET', 'ORDER_TYPE_LIMIT', 'ORDER_TYPE_STOP', 'ORDER_TYPE_STOP_LIMIT',
    'ORDER_SIDE_BUY', 'ORDER_SIDE_SELL',
    'TIMEFRAME_1M', 'TIMEFRAME_5M', 'TIMEFRAME_15M', 'TIMEFRAME_30M',
    'TIMEFRAME_1H', 'TIMEFRAME_4H', 'TIMEFRAME_1D', 'TIMEFRAME_1W', 'TIMEFRAME_1MONTH',
    'HybridModel',
    'BacktestService',
    'DataService',
    'ExecutionService'
] 