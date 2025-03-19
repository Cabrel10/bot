import pytest
import numpy as np
from datetime import datetime, timedelta

from src.strategies.trend_following import (
    TrendFollowingStrategy,
    TrendFollowingParameters
)
from src.core.data_types import MarketData
from src.core.position import PositionManager
from src.core.risk import RiskManager

@pytest.fixture
def strategy():
    params = TrendFollowingParameters(
        ema_fast=20,
        ema_slow=50,
        atr_period=14,
        adx_period=14,
        adx_threshold=25,
        risk_per_trade=0.02,
        trailing_stop=2.0
    )
    return TrendFollowingStrategy(
        position_manager=PositionManager(initial_capital=100000),
        risk_manager=RiskManager(),
        parameters=params
    )

@pytest.fixture
def market_data():
    n_points = 200
    timestamps = [
        datetime.now() + timedelta(minutes=i) 
        for i in range(n_points)
    ]
    
    # Création d'une tendance haussière
    close_prices = np.linspace(100, 150, n_points) + np.random.normal(0, 1, n_points)
    
    return MarketData(
        timestamp=timestamps,
        open=close_prices - 0.5,
        high=close_prices + 1,
        low=close_prices - 1,
        close=close_prices,
        volume=np.random.random(n_points) * 1000
    )

def test_strategy_initialization(strategy):
    assert isinstance(strategy.parameters, TrendFollowingParameters)
    assert strategy.min_required_bars > 50

def test_signal_generation(strategy, market_data):
    signals = strategy.generate_signals(market_data)
    assert isinstance(signals, list)
    
    # Vérifie si des signaux sont générés
    if signals:
        signal = signals[0]
        assert signal.direction in [-1, 1]
        assert signal.stop_loss is not None
        assert signal.size > 0
        assert 'atr' in signal.metadata

def test_position_size_calculation(strategy):
    entry_price = 100
    stop_loss = 98
    size = strategy._calculate_position_size(entry_price, stop_loss)
    
    # Vérifie que la taille respecte le risque maximum
    risk_amount = size * abs(entry_price - stop_loss)
    max_risk = strategy.position_manager.get_capital() * strategy.parameters.risk_per_trade
    assert abs(risk_amount - max_risk) < 0.01

def test_trailing_stop_update(strategy, market_data):
    # Crée une position fictive
    position = strategy.position_manager.create_position(
        direction=1,
        entry_price=100,
        size=1,
        stop_loss=95
    )
    
    # Met à jour le stop avec un nouveau prix plus haut
    strategy._update_trailing_stops(110, 5)
    
    # Vérifie que le stop a été relevé
    assert position.stop_loss > 95

def test_trend_detection(strategy, market_data):
    signals = strategy.generate_signals(market_data)
    
    # Vérifie la cohérence des signaux avec la tendance
    if signals:
        # Dans notre fixture, nous avons une tendance haussière
        assert any(signal.direction == 1 for signal in signals)

def test_risk_management(strategy):
    entry_price = 100
    stop_loss = 95
    size = strategy._calculate_position_size(entry_price, stop_loss)
    
    # Vérifie que le risque ne dépasse pas la limite
    max_loss = size * abs(entry_price - stop_loss)
    assert max_loss <= strategy.position_manager.get_capital() * strategy.parameters.risk_per_trade 