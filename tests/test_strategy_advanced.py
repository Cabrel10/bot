import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
from typing import List, Dict, Optional

from src.strategies.base_strategy import BaseStrategy, Position, Signal
from src.core.data_types import MarketData, PerformanceMetrics
from src.core.exceptions import InvalidDataError, InsufficientDataError, StrategyError
from src.core.position import PositionManager
from src.core.risk import RiskManager

class TestAdvancedStrategy(BaseStrategy):
    """Classe de stratégie pour les tests"""
    def __init__(self):
        super().__init__(
            parameters={
                'risk_per_trade': 0.02,
                'stop_loss': 0.05,
                'take_profit': 0.1,
                'trailing_stop_pct': 0.02,
                'max_positions': 3
            },
            transaction_fee=0.001
        )
        self._initialize_indicators()
    
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        return []
    
    def calculate_stop_loss(self, position: Position, market_data: MarketData) -> float:
        return position.entry_price * 0.95
    
    def calculate_take_profit(self, position: Position, market_data: MarketData) -> float:
        return position.entry_price * 1.05
    
    def calculate_position_size(self, signal: Signal, market_data: MarketData, portfolio_value: float) -> float:
        return portfolio_value * self.parameters.get('risk_per_trade', 0.02) * signal.confidence
    
    def _initialize_indicators(self) -> None:
        """Initialise les indicateurs techniques"""
        self.sma = None
        self.volatility = None
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Génère un signal de trading basé sur les données"""
        return Signal(action='HOLD', size=0.0, confidence=0.0)
    
    def update_state(self, new_data: pd.DataFrame) -> None:
        """Met à jour l'état interne de la stratégie"""
        pass

@pytest.fixture
def market_data():
    """Fixture pour générer des données de marché de test"""
    n_points = 252  # Un an de données journalières
    timestamps = [
        datetime.now() + timedelta(days=i)
        for i in range(n_points)
    ]
    
    # Génération de prix avec une tendance
    base_price = 100
    trend = np.linspace(0, 20, n_points)
    noise = np.random.normal(0, 1, n_points)
    close_prices = base_price + trend + noise
    
    return MarketData(
        timestamp=timestamps,
        open=close_prices,
        high=close_prices + 1,
        low=close_prices - 1,
        close=close_prices,
        volume=np.random.random(n_points) * 1000
    )

@pytest.fixture
def strategy():
    """Fixture pour créer une instance de stratégie de test"""
    return TestAdvancedStrategy()

def test_sharpe_ratio_calculation(strategy, market_data):
    """Test du calcul du ratio de Sharpe"""
    # Création de rendements simulés
    returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    
    # Test avec des rendements normaux
    sharpe = strategy._calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    
    # Test avec un seul rendement
    single_return = np.array([0.01])
    assert strategy._calculate_sharpe_ratio(single_return) == 0.0
    
    # Test avec des rendements constants (écart-type nul)
    constant_returns = np.array([0.01, 0.01, 0.01])
    assert strategy._calculate_sharpe_ratio(constant_returns) == 0.0

def test_transaction_cost_calculation(strategy):
    """Test du calcul des frais de transaction"""
    position_size = 100
    price = 50
    expected_cost = position_size * price * strategy.transaction_fee
    
    cost = strategy.calculate_transaction_cost(position_size, price)
    assert cost == expected_cost
    assert cost > 0

def test_position_update_with_fees(strategy):
    """Test de la mise à jour des positions avec frais"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now()
    )
    
    current_price = 51000
    strategy.update_position_with_fees(position, current_price)
    
    # Vérification que les frais sont bien pris en compte
    assert position.pnl < (current_price - position.entry_price) * position.size
    assert position.pnl_pct > 0  # Le prix a augmenté

def test_parameter_validation(strategy):
    """Test de la validation des paramètres"""
    valid_params = {
        'risk_per_trade': 0.02,
        'stop_loss': 0.05,
        'take_profit': 2.0,
        'max_positions': 3
    }
    assert strategy.validate_parameters(valid_params)
    
    # Test avec des paramètres invalides
    with pytest.raises(ValueError):
        invalid_params = valid_params.copy()
        invalid_params['risk_per_trade'] = 1.5  # Supérieur à 1
        strategy.validate_parameters(invalid_params)
    
    with pytest.raises(ValueError):
        missing_params = {'risk_per_trade': 0.02}  # Paramètres manquants
        strategy.validate_parameters(missing_params)

def test_position_sizing(strategy):
    """Test du calcul de la taille des positions"""
    signal = Signal(
        action='BUY',
        size=1.0,
        confidence=0.8
    )
    market_data = Mock()
    portfolio_value = 10000
    
    position_size = strategy.calculate_position_size(signal, market_data, portfolio_value)
    assert position_size > 0
    assert position_size <= portfolio_value

def test_stop_loss_calculation(strategy):
    """Test du calcul du stop loss"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now()
    )
    market_data = Mock()
    
    stop_loss = strategy.calculate_stop_loss(position, market_data)
    assert stop_loss < position.entry_price
    assert stop_loss > 0

def test_take_profit_calculation(strategy):
    """Test du calcul du take profit"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now()
    )
    market_data = Mock()
    
    take_profit = strategy.calculate_take_profit(position, market_data)
    assert take_profit > position.entry_price

def test_trailing_stop_update(strategy):
    """Test de la mise à jour du stop loss trailing"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now(),
        stop_loss=49000
    )
    
    # Configuration du trailing stop
    strategy.parameters['trailing_stop_pct'] = 0.02
    
    # Test avec un nouveau plus haut
    current_price = 51000
    strategy.update_trailing_stop(position, current_price)
    assert position.stop_loss > 49000  # Le stop loss devrait avoir été relevé
    
    # Test avec un prix plus bas
    current_price = 50500
    old_stop = position.stop_loss
    strategy.update_trailing_stop(position, current_price)
    assert position.stop_loss == old_stop  # Le stop loss ne devrait pas avoir changé

def test_simulate_trades(strategy, market_data):
    """Test de la simulation des trades"""
    # Création de signaux de test
    signals = [
        Signal(action='BUY', size=1.0, confidence=0.8),
        Signal(action='HOLD', size=0.0, confidence=0.5),
        Signal(action='SELL', size=1.0, confidence=0.9)
    ]
    
    # Simulation des trades
    returns = strategy._simulate_trades(signals, market_data)
    
    assert isinstance(returns, np.ndarray)
    assert len(returns) > 0
    assert all(isinstance(r, float) for r in returns)

def test_metrics_calculation(strategy):
    """Test du calcul des métriques de performance"""
    # Ajout de positions simulées
    strategy.position_manager.positions = [
        Position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000,
            entry_time=datetime.now(),
            exit_price=51000,
            exit_time=datetime.now() + timedelta(hours=1),
            pnl=1000,
            pnl_pct=0.02
        ),
        Position(
            symbol="ETH/USDT",
            size=2.0,
            entry_price=3000,
            entry_time=datetime.now(),
            exit_price=2900,
            exit_time=datetime.now() + timedelta(hours=2),
            pnl=-200,
            pnl_pct=-0.033
        )
    ]
    
    metrics = strategy.calculate_metrics()
    
    assert isinstance(metrics, dict)
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics
    assert 'avg_profit' in metrics
    assert metrics['total_trades'] == 2
    assert 0 <= metrics['win_rate'] <= 1

def test_advanced_position_management(strategy):
    """Test de la gestion avancée des positions"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now(),
        stop_loss=49000,
        take_profit=52000,
        transaction_fee=0.001
    )
    
    # Test de la mise à jour du trailing stop
    current_price = 51000
    strategy.update_trailing_stop(position, current_price)
    assert hasattr(position, 'highest_price')
    assert position.highest_price == 51000
    assert position.stop_loss > 49000
    
    # Test de la décision de fermeture
    assert not strategy.should_close_position(position, 50500)
    assert strategy.should_close_position(position, 52100)
    assert strategy.should_close_position(position, 48900)

def test_risk_management(strategy):
    """Test de la gestion des risques"""
    # Configuration des paramètres de risque
    strategy.parameters.update({
        'max_position_size': 0.1,
        'max_risk_per_trade': 0.02,
        'max_drawdown': 0.1
    })
    
    # Test du calcul de la taille de position
    signal = Signal(action='BUY', size=1.0, confidence=0.8)
    portfolio_value = 100000
    position_size = strategy.calculate_position_size(signal, Mock(), portfolio_value)
    
    assert position_size > 0
    assert position_size <= portfolio_value * strategy.parameters['max_position_size']

def test_transaction_fees_impact(strategy):
    """Test de l'impact des frais de transaction"""
    position = Position(
        symbol="BTC/USDT",
        size=1.0,
        entry_price=50000,
        entry_time=datetime.now(),
        transaction_fee=0.001  # 0.1% de frais
    )
    
    # Calcul du PnL avec frais
    exit_price = 51000
    strategy.update_position_with_fees(position, exit_price)
    
    # Vérification que les frais sont bien pris en compte
    expected_gross_pnl = (exit_price - position.entry_price) * position.size
    expected_fees = (position.entry_price + exit_price) * position.size * position.transaction_fee
    expected_net_pnl = expected_gross_pnl - expected_fees
    
    assert position.pnl < expected_gross_pnl
    assert abs(position.pnl - expected_net_pnl) < 0.01

def test_market_state_updates(strategy, market_data):
    """Test des mises à jour de l'état du marché"""
    strategy._update_market_state(market_data)
    
    assert 'current_price' in strategy.market_state
    assert 'current_volume' in strategy.market_state
    assert 'timestamp' in strategy.market_state
    assert 'volatility' in strategy.market_state
    assert 'trend' in strategy.market_state
    
    assert isinstance(strategy.market_state['volatility'], float)
    assert strategy.market_state['trend'] in [-1, 0, 1]

def test_performance_metrics_calculation(strategy):
    """Test du calcul des métriques de performance avancées"""
    # Simulation de l'historique des trades
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    
    # Test du ratio de Sharpe
    sharpe = strategy._calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)
    
    # Test avec différents taux sans risque
    sharpe_zero = strategy._calculate_sharpe_ratio(returns, risk_free_rate=0)
    sharpe_high = strategy._calculate_sharpe_ratio(returns, risk_free_rate=0.05)
    assert sharpe_zero > sharpe_high  # Le ratio devrait être plus élevé avec un taux sans risque plus bas

def test_error_handling(strategy, market_data):
    """Test de la gestion des erreurs"""
    # Test avec des données invalides
    invalid_data = MarketData(
        timestamp=[],
        open=[],
        high=[],
        low=[],
        close=[],
        volume=[]
    )
    
    with pytest.raises(InsufficientDataError):
        strategy.validate_data(invalid_data)
    
    # Test avec des paramètres invalides
    with pytest.raises(ValueError):
        strategy.parameters['risk_per_trade'] = 1.5
        strategy.calculate_position_size(
            Signal(action='BUY', size=1.0, confidence=0.8),
            market_data,
            10000
        ) 