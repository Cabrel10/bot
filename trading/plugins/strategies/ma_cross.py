"""
Module de stratégie de trading basée sur le croisement de moyennes mobiles.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class MovingAverageConfig:
    """Configuration d'une moyenne mobile."""
    type: str  # 'SMA' ou 'EMA'
    period: int

@dataclass
class MACrossConfig:
    """Configuration de la stratégie de croisement de moyennes mobiles."""
    fast_ma: MovingAverageConfig
    slow_ma: MovingAverageConfig
    min_volume: float = 1000.0

class MovingAverage:
    """Classe utilitaire pour le calcul des moyennes mobiles."""
    
    def __init__(self, config: MovingAverageConfig):
        """
        Initialise la moyenne mobile.
        
        Args:
            config: Configuration de la moyenne mobile
        """
        self.config = config
        self.values: List[float] = []
        
    def calculate(self, prices: List[float]) -> float:
        """
        Calcule la valeur de la moyenne mobile.
        
        Args:
            prices: Liste des prix
            
        Returns:
            float: Valeur de la moyenne mobile
        """
        if len(prices) < self.config.period:
            return prices[-1]
            
        if self.config.type == 'SMA':
            ma = sum(prices[-self.config.period:]) / self.config.period
        else:  # EMA
            alpha = 2 / (self.config.period + 1)
            if len(self.values) == 0:
                ma = sum(prices[-self.config.period:]) / self.config.period
            else:
                ma = (prices[-1] - self.values[-1]) * alpha + self.values[-1]
                
        self.values.append(ma)
        return ma

class MACrossStrategy:
    """Stratégie de trading basée sur le croisement de moyennes mobiles."""
    
    def __init__(self, config: MACrossConfig):
        """
        Initialise la stratégie.
        
        Args:
            config: Configuration de la stratégie
        """
        self.config = config
        self.fast_ma = MovingAverage(config.fast_ma)
        self.slow_ma = MovingAverage(config.slow_ma)
        self.positions: List[Dict[str, float]] = []
        self.last_signal: Optional[str] = None
        
    def calculate_signals(self, candles: List[Dict[str, float]]) -> Dict[str, any]:
        """
        Calcule les signaux de trading basés sur les croisements de moyennes mobiles.
        
        Args:
            candles: Liste des bougies (OHLCV)
            
        Returns:
            Dict[str, any]: Signaux de trading et informations supplémentaires
        """
        if len(candles) < max(self.config.fast_ma.period, self.config.slow_ma.period):
            return {
                'signal': None,
                'fast_ma': None,
                'slow_ma': None,
                'volume_ok': False
            }
        
        # Extraction des prix de clôture et volumes
        closes = [c['close'] for c in candles]
        volumes = [c['volume'] for c in candles]
        
        # Calcul des moyennes mobiles
        fast_ma = self.fast_ma.calculate(closes)
        slow_ma = self.slow_ma.calculate(closes)
        
        # Vérification du volume
        volume_ok = volumes[-1] >= self.config.min_volume
        
        # Détermination du signal
        signal = None
        if len(self.fast_ma.values) >= 2 and len(self.slow_ma.values) >= 2:
            # Croisement haussier
            if (self.fast_ma.values[-2] <= self.slow_ma.values[-2] and
                self.fast_ma.values[-1] > self.slow_ma.values[-1]):
                signal = 'buy'
            # Croisement baissier
            elif (self.fast_ma.values[-2] >= self.slow_ma.values[-2] and
                  self.fast_ma.values[-1] < self.slow_ma.values[-1]):
                signal = 'sell'
        
        # Mise à jour du dernier signal
        if signal and volume_ok:
            self.last_signal = signal
        
        return {
            'signal': signal if volume_ok else None,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'volume_ok': volume_ok
        }
    
    def should_open_position(self, signal: Dict[str, any], current_positions: List[Dict]) -> bool:
        """
        Détermine si une position doit être ouverte.
        
        Args:
            signal: Signaux de trading
            current_positions: Positions actuellement ouvertes
            
        Returns:
            bool: True si une position doit être ouverte
        """
        if not signal['signal'] or not signal['volume_ok']:
            return False
            
        # Vérifie si nous avons déjà une position
        if current_positions:
            return False
            
        # Vérifie si le signal est différent du dernier
        return signal['signal'] != self.last_signal
    
    def should_close_position(self, signal: Dict[str, any], position: Dict) -> bool:
        """
        Détermine si une position doit être fermée.
        
        Args:
            signal: Signaux de trading
            position: Position à évaluer
            
        Returns:
            bool: True si la position doit être fermée
        """
        if not signal['signal'] or not signal['volume_ok']:
            return False
            
        # Ferme la position si le signal est opposé à la position
        return ((position['side'] == 'long' and signal['signal'] == 'sell') or
                (position['side'] == 'short' and signal['signal'] == 'buy'))
    
    def get_position_size(self, capital: float, price: float) -> float:
        """
        Calcule la taille de la position à prendre.
        
        Args:
            capital: Capital disponible
            price: Prix actuel
            
        Returns:
            float: Taille de la position
        """
        # Utilise 2% du capital par position
        position_value = capital * 0.02
        return position_value / price
    
    def get_stop_loss(self, signal: Dict[str, any], entry_price: float) -> float:
        """
        Calcule le niveau de stop loss.
        
        Args:
            signal: Signaux de trading
            entry_price: Prix d'entrée
            
        Returns:
            float: Niveau de stop loss
        """
        # Stop loss à 2% du prix d'entrée
        if signal['signal'] == 'buy':
            return entry_price * 0.98
        else:
            return entry_price * 1.02 