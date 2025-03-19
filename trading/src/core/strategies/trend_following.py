from typing import List, Dict, Optional
import numpy as np
import logging
from dataclasses import dataclass
import pandas as pd

from trading.strategies.base_strategy import BaseStrategy, Signal
from trading.core.data_types import MarketData, StrategyParameters
from trading.indicators.feature_engineer import (
    calculate_ema,
    calculate_atr,
    calculate_adx
)
from trading.core.position import PositionManager
from trading.core.risk import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class TrendFollowingParameters(StrategyParameters):
    """Paramètres spécifiques à la stratégie de suivi de tendance"""
    ema_fast: int = 20        # Période EMA rapide
    ema_slow: int = 50        # Période EMA lente
    atr_period: int = 14      # Période ATR pour la volatilité
    adx_period: int = 14      # Période ADX pour force de la tendance
    adx_threshold: float = 25  # Seuil ADX minimum pour confirmer la tendance
    risk_per_trade: float = 0.02  # Risque maximum par trade (2% du capital)
    trailing_stop: float = 2.0  # Stop trailing en multiples d'ATR

class TrendFollowingStrategy(BaseStrategy):
    """
    Stratégie de suivi de tendance basée sur les EMA et l'ADX.
    
    Logique:
    - Entrée: Croisement des EMA avec confirmation ADX
    - Sortie: Croisement inverse ou stop trailing
    - Taille position: Basée sur l'ATR et le risque maximum par trade
    """

    def __init__(self, config: Dict):
        """
        Initialise la stratégie de suivi de tendance.
        
        Args:
            config: Configuration de la stratégie contenant:
                - fast_period: Période pour la moyenne mobile rapide
                - slow_period: Période pour la moyenne mobile lente
                - atr_period: Période pour l'ATR
                - atr_multiplier: Multiplicateur pour les stops
                - adx_period: Période pour l'ADX
                - adx_threshold: Seuil pour l'ADX
        """
        super().__init__(config)
        self.fast_period = config.get('fast_period', 20)
        self.slow_period = config.get('slow_period', 50)
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        
        self.min_required_bars = max(
            self.slow_period,
            parameters.ema_slow,
            parameters.atr_period,
            parameters.adx_period
        ) + 10

        logger.info(f"Initialisation {self.name} avec paramètres: {parameters}")

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère les signaux de trading basés sur les croisements d'EMA et l'ADX.
        
        Args:
            market_data: Données de marché OHLCV
            
        Returns:
            List[Signal]: Liste des signaux générés
        """
        try:
            self.validate_data(market_data)
            
            # Calcul des indicateurs
            ema_fast = calculate_ema(market_data.close, self.parameters.ema_fast)
            ema_slow = calculate_ema(market_data.close, self.parameters.ema_slow)
            atr = calculate_atr(market_data, self.parameters.atr_period)
            adx = calculate_adx(market_data, self.parameters.adx_period)
            
            signals = []
            
            # Dernière barre complète
            current_idx = -1
            
            # Vérification des conditions de signal
            trend_strength = adx[current_idx] > self.parameters.adx_threshold
            
            if trend_strength:
                # Croisement haussier
                if (ema_fast[current_idx-1] <= ema_slow[current_idx-1] and 
                    ema_fast[current_idx] > ema_slow[current_idx]):
                    
                    signal = self._create_long_signal(
                        timestamp=market_data.timestamp[current_idx],
                        price=market_data.close[current_idx],
                        atr=atr[current_idx]
                    )
                    signals.append(signal)
                
                # Croisement baissier
                elif (ema_fast[current_idx-1] >= ema_slow[current_idx-1] and 
                      ema_fast[current_idx] < ema_slow[current_idx]):
                    
                    signal = self._create_short_signal(
                        timestamp=market_data.timestamp[current_idx],
                        price=market_data.close[current_idx],
                        atr=atr[current_idx]
                    )
                    signals.append(signal)
            
            # Mise à jour des stops trailing
            self._update_trailing_stops(market_data.close[current_idx], atr[current_idx])
            
            return signals
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {str(e)}")
            return []

    def _create_long_signal(self, timestamp, price: float, atr: float) -> Signal:
        """Crée un signal d'achat avec stop et taille de position."""
        stop_loss = price - (atr * self.parameters.trailing_stop)
        position_size = self._calculate_position_size(price, stop_loss)
        
        return Signal(
            timestamp=timestamp,
            direction=1,
            strength=1.0,
            price=price,
            stop_loss=stop_loss,
            size=position_size,
            metadata={
                'atr': atr,
                'type': 'trend_following_long'
            }
        )

    def _create_short_signal(self, timestamp, price: float, atr: float) -> Signal:
        """Crée un signal de vente avec stop et taille de position."""
        stop_loss = price + (atr * self.parameters.trailing_stop)
        position_size = self._calculate_position_size(price, stop_loss)
        
        return Signal(
            timestamp=timestamp,
            direction=-1,
            strength=1.0,
            price=price,
            stop_loss=stop_loss,
            size=position_size,
            metadata={
                'atr': atr,
                'type': 'trend_following_short'
            }
        )

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calcule la taille de la position basée sur le risque par trade.
        """
        risk_amount = self.position_manager.get_capital() * self.parameters.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        return risk_amount / risk_per_unit

    def _update_trailing_stops(self, current_price: float, current_atr: float) -> None:
        """
        Met à jour les stops trailing pour les positions ouvertes.
        """
        active_positions = self.position_manager.get_active_positions()
        
        for position in active_positions:
            if position.direction == 1:  # Long position
                new_stop = current_price - (current_atr * self.parameters.trailing_stop)
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:  # Short position
                new_stop = current_price + (current_atr * self.parameters.trailing_stop)
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop 