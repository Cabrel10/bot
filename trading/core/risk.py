"""
Risk management module.
"""

from typing import Dict, List, Optional
from datetime import datetime
from .data_types import MarketData, PositionData

class RiskManager:
    """Manages trading risks and position sizing."""
    
    def __init__(
        self,
        max_position_size: float,
        max_drawdown: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        max_leverage: float = 1.0
    ):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_leverage = max_leverage
        self._positions: Dict[str, PositionData] = {}
        self._equity_curve: List[float] = []
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float
    ) -> float:
        """Calculate position size based on risk parameters."""
        risk_amount = capital * self.max_drawdown
        position_size = risk_amount / (price * volatility)
        return min(position_size, self.max_position_size)
    
    def update_position_risk(
        self,
        position: PositionData,
        current_price: float
    ) -> Dict[str, float]:
        """Update risk metrics for a position."""
        entry_price = position.average_price
        current_pnl = (current_price - entry_price) / entry_price
        
        return {
            'unrealized_pnl': current_pnl,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'take_profit': entry_price * (1 + self.take_profit_pct)
        }
    
    def check_risk_limits(
        self,
        position: PositionData,
        market_data: MarketData
    ) -> bool:
        """Check if position violates risk limits."""
        risk_metrics = self.update_position_risk(position, market_data.close)
        
        # Check stop loss
        if market_data.low <= risk_metrics['stop_loss']:
            return False
            
        # Check take profit
        if market_data.high >= risk_metrics['take_profit']:
            return False
            
        # Check drawdown
        if risk_metrics['unrealized_pnl'] <= -self.max_drawdown:
            return False
            
        return True
    
    def update_equity_curve(
        self,
        timestamp: datetime,
        equity: float
    ) -> None:
        """Update equity curve with new value."""
        self._equity_curve.append(equity)
    
    def get_drawdown(self) -> float:
        """Calculate current drawdown."""
        if not self._equity_curve:
            return 0.0
            
        peak = max(self._equity_curve)
        current = self._equity_curve[-1]
        return (peak - current) / peak
    
    def get_position_risk(
        self,
        symbol: str
    ) -> Optional[Dict[str, float]]:
        """Get risk metrics for a specific position."""
        position = self._positions.get(symbol)
        if not position:
            return None
            
        return self.update_position_risk(position, position.average_price)
    
    def set_position(
        self,
        symbol: str,
        position: PositionData
    ) -> None:
        """Set or update a position."""
        self._positions[symbol] = position
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        if symbol in self._positions:
            del self._positions[symbol] 