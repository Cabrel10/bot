"""
Position management module.
"""

from typing import Dict, List, Optional
from datetime import datetime
from .data_types import PositionData, OrderData, TradeData

class PositionManager:
    """Manages trading positions and orders."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._positions: Dict[str, PositionData] = {}
        self._orders: Dict[str, OrderData] = {}
        self._trades: List[TradeData] = []
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> PositionData:
        """Open a new position."""
        position = PositionData(
            symbol=symbol,
            quantity=quantity,
            average_price=price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=datetime.now()
        )
        self._positions[symbol] = position
        return position
    
    def close_position(
        self,
        symbol: str,
        price: float
    ) -> Optional[TradeData]:
        """Close an existing position."""
        position = self._positions.get(symbol)
        if not position:
            return None
            
        pnl = (price - position.average_price) * position.quantity
        if position.side == 'sell':
            pnl = -pnl
            
        trade = TradeData(
            trade_id=f"trade_{len(self._trades)}",
            order_id=f"order_{len(self._orders)}",
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            price=price,
            timestamp=datetime.now(),
            fee=0.0
        )
        
        self._trades.append(trade)
        self.current_capital += pnl
        del self._positions[symbol]
        
        return trade
    
    def update_position(
        self,
        symbol: str,
        current_price: float
    ) -> None:
        """Update position with current market price."""
        position = self._positions.get(symbol)
        if not position:
            return
            
        position.unrealized_pnl = (
            (current_price - position.average_price) * position.quantity
        )
        if position.side == 'sell':
            position.unrealized_pnl = -position.unrealized_pnl
    
    def get_position(self, symbol: str) -> Optional[PositionData]:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionData]:
        """Get all open positions."""
        return self._positions.copy()
    
    def get_position_value(
        self,
        symbol: str,
        current_price: float
    ) -> float:
        """Calculate current value of a position."""
        position = self._positions.get(symbol)
        if not position:
            return 0.0
        return position.quantity * current_price
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = self.current_capital
        for symbol, position in self._positions.items():
            if symbol in prices:
                total += self.get_position_value(symbol, prices[symbol])
        return total
    
    def add_order(self, order: OrderData) -> None:
        """Add a new order."""
        self._orders[order.order_id] = order
    
    def remove_order(self, order_id: str) -> None:
        """Remove an order."""
        if order_id in self._orders:
            del self._orders[order_id]
    
    def get_order(self, order_id: str) -> Optional[OrderData]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_all_orders(self) -> Dict[str, OrderData]:
        """Get all active orders."""
        return self._orders.copy()
    
    def get_trade_history(self) -> List[TradeData]:
        """Get complete trade history."""
        return self._trades.copy()
