from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass

from trading.core.exceptions import (
    InvalidDataError,
    StrategyError,
    InsufficientDataError,
    PositionError
)
from trading.core.data_types import (
    MarketData,
    Signal,
    Position,
    StrategyParameters,
    PerformanceMetrics
)
from trading.core.position import PositionManager
from trading.core.risk import RiskManager
from trading.utils.validation import validate_numeric_range
from trading.utils.logger import TradingLogger

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Représente une position de trading"""
    symbol: str
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Signal:
    """Signal de trading généré par une stratégie"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    size: float
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict] = None

class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.
    S'intègre avec PositionManager et RiskManager pour une gestion complète.
    
    Attributes:
        name (str): Nom de la stratégie
        position_manager (PositionManager): Gestionnaire des positions
        risk_manager (RiskManager): Gestionnaire des risques
        parameters (StrategyParameters): Paramètres de la stratégie
        market_state (Dict): État actuel du marché
    """

    def __init__(self, 
                 position_manager: Optional[PositionManager] = None,
                 risk_manager: Optional[RiskManager] = None,
                 parameters: Optional[StrategyParameters] = None):
        """
        Initialise la stratégie avec ses gestionnaires.

        Args:
            position_manager: Gestionnaire des positions
            risk_manager: Gestionnaire des risques
            parameters: Paramètres initiaux de la stratégie
        """
        self.name: str = self.__class__.__name__
        self.position_manager = position_manager or PositionManager()
        self.risk_manager = risk_manager or RiskManager()
        self.parameters = parameters or {}
        self.market_state: Dict = {}
        self.min_required_bars: int = 100
        
        # Métriques de performance en temps réel
        self.current_metrics: PerformanceMetrics = PerformanceMetrics()
        
        logger.info(f"Initialisation de la stratégie {self.name}")

    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Génère des signaux de trading basés sur les données de marché.
        Doit être implémentée par les classes filles.
        """
        raise NotImplementedError

    def update(self, market_data: MarketData) -> Tuple[List[Signal], List[Position]]:
        """
        Met à jour la stratégie avec les nouvelles données de marché.
        
        Args:
            market_data: Nouvelles données de marché
            
        Returns:
            Tuple[List[Signal], List[Position]]: Signaux générés et positions mises à jour
            
        Raises:
            StrategyError: En cas d'erreur durant la mise à jour
        """
        try:
            self.validate_data(market_data)
            self._update_market_state(market_data)
            
            # Génération des signaux
            signals = self.generate_signals(market_data)
            
            # Validation des signaux par le risk manager
            validated_signals = [
                signal for signal in signals
                if self.risk_manager.validate_signal(signal, market_data)
            ]
            
            # Mise à jour des positions
            updated_positions = self.position_manager.update_positions(
                validated_signals,
                market_data
            )
            
            # Mise à jour des métriques
            self._update_metrics(updated_positions)
            
            return validated_signals, updated_positions
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la stratégie: {str(e)}")
            raise StrategyError(f"Erreur de mise à jour: {str(e)}")

    def _update_market_state(self, market_data: MarketData) -> None:
        """
        Met à jour l'état du marché avec les dernières données.
        """
        latest_idx = -1
        self.market_state.update({
            'current_price': market_data.close[latest_idx],
            'current_volume': market_data.volume[latest_idx],
            'timestamp': market_data.timestamp[latest_idx],
            'volatility': self._calculate_volatility(market_data.close),
            'trend': self._detect_trend(market_data.close)
        })

    def _calculate_volatility(self, prices: np.ndarray, window: int = 20) -> float:
        """
        Calcule la volatilité sur une fenêtre donnée.
        """
        if len(prices) < window:
            return 0.0
        returns = np.log(prices[1:] / prices[:-1])
        return float(np.std(returns[-window:]) * np.sqrt(252))

    def _detect_trend(self, prices: np.ndarray, window: int = 20) -> int:
        """
        Détecte la tendance actuelle (-1: baissière, 0: neutre, 1: haussière).
        """
        if len(prices) < window:
            return 0
        sma = np.mean(prices[-window:])
        current_price = prices[-1]
        return int(np.sign(current_price - sma))

    def _update_metrics(self, positions: List[Position]) -> None:
        """
        Met à jour les métriques de performance en temps réel.
        """
        if not positions:
            return
            
        closed_positions = [p for p in positions if p.is_closed]
        if not closed_positions:
            return
            
        returns = [pos.pnl_pct for pos in closed_positions]
        self.current_metrics.total_return = sum(returns)
        self.current_metrics.win_rate = len([r for r in returns if r > 0]) / len(returns)
        self.current_metrics.sharpe_ratio = self._calculate_sharpe_ratio(np.array(returns))
        self.current_metrics.number_of_trades = len(closed_positions)

    def validate_data(self, market_data: MarketData) -> bool:
        """
        Valide les données de marché.
        """
        try:
            if not isinstance(market_data, MarketData):
                raise InvalidDataError("Les données doivent être de type MarketData")
                
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if not hasattr(market_data, field) or getattr(market_data, field) is None:
                    raise InvalidDataError(f"Champ requis manquant: {field}")
                    
            if len(market_data.close) < self.min_required_bars:
                raise InsufficientDataError(
                    f"Minimum {self.min_required_bars} barres requises, "
                    f"reçu {len(market_data.close)}"
                )
                
            # Validation des valeurs
            validate_numeric_range(market_data.close, "Prix de clôture", min_value=0)
            validate_numeric_range(market_data.volume, "Volume", min_value=0)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur de validation des données: {str(e)}")
            raise

    def get_state(self) -> Dict:
        """
        Retourne l'état actuel de la stratégie.
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'market_state': self.market_state,
            'current_metrics': self.current_metrics.__dict__,
            'active_positions': self.position_manager.get_active_positions(),
            'risk_limits': self.risk_manager.get_current_limits()
        }

    @abstractmethod
    def _initialize_indicators(self) -> None:
        """Initialise les indicateurs techniques utilisés par la stratégie"""
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Génère un signal de trading basé sur les données.
        
        Args:
            data: DataFrame avec les données OHLCV et indicateurs
            
        Returns:
            Signal: Signal de trading généré
        """
        pass
    
    @abstractmethod
    def update_state(self, new_data: pd.DataFrame) -> None:
        """
        Met à jour l'état interne de la stratégie.
        
        Args:
            new_data: Nouvelles données de marché
        """
        pass
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calcule la taille de la position à prendre.
        
        Args:
            signal: Signal de trading
            portfolio_value: Valeur actuelle du portefeuille
            
        Returns:
            float: Taille de la position
        """
        risk_per_trade = self.parameters.get('risk_per_trade', 0.02)
        return portfolio_value * risk_per_trade * signal.confidence
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """
        Détermine si une position doit être fermée.
        
        Args:
            position: Position à évaluer
            current_price: Prix actuel
            
        Returns:
            bool: True si la position doit être fermée
        """
        if position.stop_loss and current_price <= position.stop_loss:
            return True
        if position.take_profit and current_price >= position.take_profit:
            return True
        return False
    
    def update_stops(self, position: Position, current_price: float) -> None:
        """
        Met à jour les stops de la position.
        
        Args:
            position: Position à mettre à jour
            current_price: Prix actuel
        """
        trailing_stop = self.parameters.get('trailing_stop_pct')
        if trailing_stop and position.size > 0:  # Position longue
            new_stop = current_price * (1 - trailing_stop)
            if not position.stop_loss or new_stop > position.stop_loss:
                position.stop_loss = new_stop
        elif trailing_stop and position.size < 0:  # Position courte
            new_stop = current_price * (1 + trailing_stop)
            if not position.stop_loss or new_stop < position.stop_loss:
                position.stop_loss = new_stop
    
    def calculate_metrics(self) -> Dict:
        """
        Calcule les métriques de performance de la stratégie.
        
        Returns:
            Dict: Métriques de performance
        """
        return {
            'total_trades': len(self.position_manager.positions),
            'current_position': self.position_manager.current_position,
            'win_rate': self._calculate_win_rate(),
            'avg_profit': self._calculate_average_profit()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calcule le taux de réussite des trades"""
        if not self.position_manager.positions:
            return 0.0
        winning_trades = sum(1 for p in self.position_manager.positions if p.size > 0)
        return winning_trades / len(self.position_manager.positions)
    
    def _calculate_average_profit(self) -> float:
        """Calcule le profit moyen par trade"""
        if not self.position_manager.positions:
            return 0.0
        total_profit = sum(p.size * (p.entry_price - p.stop_loss if p.stop_loss else 0)
                         for p in self.position_manager.positions)
        return total_profit / len(self.position_manager.positions)
    
    def reset(self) -> None:
        """Réinitialise l'état de la stratégie"""
        self.position_manager.current_position = 0.0
        self.position_manager.positions.clear()
        self.current_metrics.total_return = 0.0
        self.current_metrics.win_rate = 0.0
        self.current_metrics.sharpe_ratio = 0.0
        self.current_metrics.number_of_trades = 0
        self._initialize_indicators()