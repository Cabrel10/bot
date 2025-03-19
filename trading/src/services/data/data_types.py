"""
Types de données pour le service de trading.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Trade:
    """Information sur un trade."""
    
    timestamp: datetime
    symbol: str
    side: str  # 'buy' ou 'sell'
    price: float
    amount: float
    cost: float = 0.0  # price * amount
    fee: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
