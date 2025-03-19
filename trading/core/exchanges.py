"""
Base exchange class and implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from .data_types import MarketData, OrderData, TradeData
from .types import (
    TIMEFRAME_1M, TIMEFRAME_5M, TIMEFRAME_15M, TIMEFRAME_30M,
    TIMEFRAME_1H, TIMEFRAME_2H, TIMEFRAME_4H, TIMEFRAME_6H,
    TIMEFRAME_8H, TIMEFRAME_12H, TIMEFRAME_1D, TIMEFRAME_3D,
    TIMEFRAME_1W, TIMEFRAME_1M_MONTH
)

class BaseExchange(ABC):
    """Base class for cryptocurrency exchanges."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = None
    
    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MarketData]:
        """Get historical kline data."""
        pass
    
    @abstractmethod
    async def get_websocket_data(
        self,
        symbol: str,
        timeframe: str
    ) -> MarketData:
        """Get real-time market data via websocket."""
        pass
    
    @abstractmethod
    async def create_order(
        self,
        order: OrderData
    ) -> TradeData:
        """Create a new order."""
        pass
    
    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
        symbol: str
    ) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(
        self,
        order_id: str,
        symbol: str
    ) -> Dict:
        """Get the status of an order."""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balances."""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get available trading pairs."""
        pass
    
    @abstractmethod
    async def get_exchange_info(self) -> Dict:
        """Get exchange information."""
        pass 