"""
Module de gestion des positions de trading.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Position:
    """Information sur une position de trading."""
    
    # Paramètres obligatoires
    symbol: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    
    # Propriétés calculées
    @property
    def profit_pct(self) -> float:
        """Calcule le pourcentage de profit/perte."""
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def profit_amount(self) -> float:
        """Calcule le montant du profit/perte."""
        return (self.exit_price - self.entry_price) * self.size
    
    @property
    def duration(self) -> float:
        """Calcule la durée de la position en heures."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600
