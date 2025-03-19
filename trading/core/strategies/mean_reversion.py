from typing import List, Dict, Optional
import numpy as np
import logging
from dataclasses import dataclass
import pandas as pd

from trading.strategies.base_strategy import BaseStrategy, Signal
from trading.core.data_types import MarketData, StrategyParameters
from trading.indicators.feature_engineer import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_atr
)
from trading.core.position import PositionManager
from trading.core.risk import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class MeanReversionParameters(StrategyParameters):
    """Paramètres de la stratégie de mean reversion"""
    rsi_period: int = 14          # Période du RSI
    rsi_overbought: float = 70    # Niveau de surachat
    rsi_oversold: float = 30      # Niveau de survente
    bb_period: int = 20           # Période des bandes de Bollinger
    bb_std: float = 2.0           # Nombre d'écarts-types pour les bandes
    atr_period: int = 14          # Période ATR pour les stops
    risk_per_trade: float = 0.01  # Risque maximum par trade (1% du capital)
    profit_target: float = 2.0    # Target en multiples de l'ATR
    stop_loss: float = 1.5        # Stop loss en multiples de l'ATR

class MeanReversionStrategy(BaseStrategy):
    """Stratégie de trading basée sur le retour à la moyenne"""
    
    def __init__(self, config: Dict):
        """
        Initialise la stratégie de retour à la moyenne.
        
        Args:
            config: Configuration de la stratégie contenant:
                - lookback_period: Période pour le calcul de la moyenne mobile
                - std_dev_threshold: Nombre d'écarts-types pour les bandes
                - rsi_period: Période pour le RSI
                - rsi_overbought: Niveau de surachat du RSI
                - rsi_oversold: Niveau de survente du RSI
        """
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 20)
        self.std_dev_threshold = config.get('std_dev_threshold', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
    
    def _initialize_indicators(self) -> None:
        """Initialise les indicateurs techniques"""
        self.sma = None
        self.upper_band = None
        self.lower_band = None
        self.rsi = None
        self.volatility = None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> None:
        """
        Calcule les indicateurs techniques.
        
        Args:
            data: DataFrame avec les données OHLCV
        """
        # Calcul de la moyenne mobile et des bandes de Bollinger
        self.sma = data['close'].rolling(window=self.lookback_period).mean()
        rolling_std = data['close'].rolling(window=self.lookback_period).std()
        self.upper_band = self.sma + (rolling_std * self.std_dev_threshold)
        self.lower_band = self.sma - (rolling_std * self.std_dev_threshold)
        
        # Calcul du RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))
        
        # Calcul de la volatilité
        self.volatility = rolling_std / self.sma
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Génère un signal de trading basé sur les indicateurs.
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Signal: Signal de trading généré
        """
        self._calculate_indicators(data)
        
        current_price = data['close'].iloc[-1]
        current_rsi = self.rsi.iloc[-1]
        
        # Calcul de la distance par rapport à la moyenne
        deviation = (current_price - self.sma.iloc[-1]) / self.sma.iloc[-1]
        
        # Calcul de la confiance du signal
        confidence = min(abs(deviation) / self.std_dev_threshold, 1.0)
        
        # Génération du signal
        if current_price > self.upper_band.iloc[-1] and current_rsi > self.rsi_overbought:
            # Signal de vente (surachat)
            return Signal(
                action='SELL',
                size=-1.0,
                confidence=confidence,
                stop_loss=current_price * (1 + self.config.get('stop_loss_pct', 0.02)),
                take_profit=self.sma.iloc[-1],
                metadata={'rsi': current_rsi, 'deviation': deviation}
            )
        elif current_price < self.lower_band.iloc[-1] and current_rsi < self.rsi_oversold:
            # Signal d'achat (survente)
            return Signal(
                action='BUY',
                size=1.0,
                confidence=confidence,
                stop_loss=current_price * (1 - self.config.get('stop_loss_pct', 0.02)),
                take_profit=self.sma.iloc[-1],
                metadata={'rsi': current_rsi, 'deviation': deviation}
            )
        else:
            # Pas de signal
            return Signal(
                action='HOLD',
                size=0.0,
                confidence=0.0,
                metadata={'rsi': current_rsi, 'deviation': deviation}
            )
    
    def update_state(self, new_data: pd.DataFrame) -> None:
        """
        Met à jour l'état de la stratégie avec les nouvelles données.
        
        Args:
            new_data: Nouvelles données de marché
        """
        self._calculate_indicators(new_data)
        
        # Mise à jour des stops pour les positions existantes
        current_price = new_data['close'].iloc[-1]
        for position in self.positions:
            self.update_stops(position, current_price)
            
            # Vérification des conditions de sortie
            if self.should_close_position(position, current_price):
                self.positions.remove(position)
                self.current_position -= position.size
    
    def _calculate_dynamic_threshold(self) -> float:
        """
        Calcule un seuil dynamique basé sur la volatilité.
        
        Returns:
            float: Seuil dynamique
        """
        if self.volatility is not None:
            return self.std_dev_threshold * (1 + self.volatility.iloc[-1])
        return self.std_dev_threshold

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère les signaux de trading basés sur le RSI et les bandes de Bollinger.
        
        Args:
            market_data: Données de marché OHLCV
            
        Returns:
            List[Signal]: Liste des signaux générés
        """
        try:
            self.validate_data(market_data)
            
            # Calcul des indicateurs
            rsi = calculate_rsi(market_data.close, self.parameters.rsi_period)
            bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(
                market_data.close,
                self.parameters.bb_period,
                self.parameters.bb_std
            )
            atr = calculate_atr(market_data, self.parameters.atr_period)
            
            signals = []
            current_idx = -1
            current_price = market_data.close[current_idx]
            
            # Vérification des conditions d'entrée
            # Signal long (survente)
            if (rsi[current_idx] < self.parameters.rsi_oversold and 
                current_price < bb_lower[current_idx]):
                
                signal = self._create_long_signal(
                    timestamp=market_data.timestamp[current_idx],
                    price=current_price,
                    atr=atr[current_idx],
                    target_price=bb_middle[current_idx]
                )
                signals.append(signal)
            
            # Signal short (surachat)
            elif (rsi[current_idx] > self.parameters.rsi_overbought and 
                  current_price > bb_upper[current_idx]):
                
                signal = self._create_short_signal(
                    timestamp=market_data.timestamp[current_idx],
                    price=current_price,
                    atr=atr[current_idx],
                    target_price=bb_middle[current_idx]
                )
                signals.append(signal)
            
            # Mise à jour des positions existantes
            self._update_positions(
                current_price=current_price,
                middle_band=bb_middle[current_idx],
                atr=atr[current_idx]
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {str(e)}")
            return []

    def _create_long_signal(self, timestamp, price: float, atr: float, 
                          target_price: float) -> Signal:
        """Crée un signal d'achat avec stop et target."""
        stop_loss = price - (atr * self.parameters.stop_loss)
        position_size = self._calculate_position_size(price, stop_loss)
        
        return Signal(
            timestamp=timestamp,
            direction=1,
            strength=1.0,
            price=price,
            stop_loss=stop_loss,
            take_profit=target_price,
            size=position_size,
            metadata={
                'atr': atr,
                'type': 'mean_reversion_long'
            }
        )

    def _create_short_signal(self, timestamp, price: float, atr: float,
                           target_price: float) -> Signal:
        """Crée un signal de vente avec stop et target."""
        stop_loss = price + (atr * self.parameters.stop_loss)
        position_size = self._calculate_position_size(price, stop_loss)
        
        return Signal(
            timestamp=timestamp,
            direction=-1,
            strength=1.0,
            price=price,
            stop_loss=stop_loss,
            take_profit=target_price,
            size=position_size,
            metadata={
                'atr': atr,
                'type': 'mean_reversion_short'
            }
        )

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calcule la taille de la position basée sur le risque."""
        risk_amount = self.position_manager.get_capital() * self.parameters.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
            
        return risk_amount / risk_per_unit

    def _update_positions(self, current_price: float, middle_band: float,
                         atr: float) -> None:
        """Met à jour les positions existantes."""
        active_positions = self.position_manager.get_active_positions()
        
        for position in active_positions:
            # Ajustement des stops si le prix se rapproche du target
            distance_to_target = abs(current_price - position.take_profit)
            initial_risk = abs(position.entry_price - position.stop_loss)
            
            if distance_to_target < initial_risk * 0.5:
                # Déplace le stop au break-even
                if position.direction == 1:  # Long
                    new_stop = max(position.entry_price, position.stop_loss)
                else:  # Short
                    new_stop = min(position.entry_price, position.stop_loss)
                    
                position.stop_loss = new_stop