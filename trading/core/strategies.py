"""
Base strategy class and implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from .data_types import MarketData, OrderData, StrategyParameters
from .position import PositionManager
from .risk import RiskManager

class BaseStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(
        self,
        parameters: StrategyParameters,
        position_manager: PositionManager,
        risk_manager: RiskManager
    ):
        self.parameters = parameters
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self._indicators: Dict[str, float] = {}
    
    @abstractmethod
    async def calculate_indicators(
        self,
        market_data: List[MarketData]
    ) -> Dict[str, float]:
        """Calculate technical indicators."""
        pass
    
    @abstractmethod
    async def generate_signals(
        self,
        market_data: List[MarketData]
    ) -> List[OrderData]:
        """Generate trading signals."""
        pass
    
    @abstractmethod
    async def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        pass
    
    @abstractmethod
    async def update_state(
        self,
        market_data: MarketData,
        position: Optional[Dict] = None
    ) -> None:
        """Update strategy state."""
        pass
    
    @abstractmethod
    async def get_position_size(
        self,
        market_data: MarketData,
        signal: OrderData
    ) -> float:
        """Calculate position size based on risk management."""
        pass
    
    @abstractmethod
    async def should_exit(
        self,
        position: Dict,
        market_data: MarketData
    ) -> bool:
        """Determine if position should be closed."""
        pass 