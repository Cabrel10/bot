import pytest
import numpy as np
from datetime import datetime, timedelta

from src.strategies.mean_reversion import (
    MeanReversionStrategy,
    MeanReversionParameters
)
from src.core.data_types import MarketData
from src.core.position import PositionManager
from src.core.risk import RiskManager

@pytest.fixture
def strategy():
    params = MeanReversionParameters(
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        risk_per_trade=0.01,
        profit_target=2.0,
        stop_loss=1.5
    )
    return MeanReversionStrategy(
        position_manager=PositionManager(initial_capital=100000),
        risk_manager=RiskManager(),
        parameters=params
    )

@pytest.fixture
def market_data_oversold():
    """Crée des données de marché en condition de survente."""
    n_points = 100
    timestamps = [datetime.now() + timedelta(minutes=i) for i in range(n_points)]
    
    # Crée une tendance baissière suivie d'une stabilisation
    close_prices = np.linspace(100, 80, n_points) + np.random.normal(0, 0.5, n_points)
    close_prices[-10:] = 80  # Stabilisation
    
    return MarketData(
        timestamp=timestamps,
        open=close_prices - 0.2,
        high=close_prices + 0.5,
        low=close_prices - 0.5,
        close=close_prices,
        volume=np.random.random(n_points) * 1000
    )

@pytest.fixture
def market_data_overbought():
    """Crée des données de marché en condition de surachat."""
    n_points = 100
    timestamps = [datetime.now() + timedelta(minutes=i) for i in range(n_points)]
    
    # Crée une tendance haussière suivie d'une stabilisation
    close_prices = np.linspace(100, 120, n_points) + np.random.normal(0, 0.5, n_points)
    close_prices[-10:] = 120  # Stabilisation
    
    return MarketData(
        timestamp=timestamps,
        open=close_prices - 0.2,
        high=close_prices + 0.5,
        low=close_prices - 0.5,
        close=close_prices,
        volume=np.random.random(n_points) * 1000
    )

def test_strategy_initialization(strategy):
    assert isinstance(strategy.parameters, MeanReversionParameters)
    assert strategy.min_required_bars > 20

def test_oversold_signal_generation(strategy, market_data_oversold):
    signals = strategy.generate_signals(market_data_oversold)
    
    # Vérifie si des signaux longs sont générés en condition de survente
    if signals:
        assert any(signal.direction == 1 for signal in signals)
        signal = next(s for s in signals if s.direction == 1)
        assert signal.stop_loss < signal.price
        assert signal.take_profit > signal.price

def test_overbought_signal_generation(strategy, market_data_overbought):
    signals = strategy.generate_signals(market_data_overbought)
    
    # Vérifie si des signaux courts sont générés en condition de surachat
    if signals:
        assert any(signal.direction == -1 for signal in signals)
        signal = next(s for s in signals if s.direction == -1)
        assert signal.stop_loss > signal.price
        assert signal.take_profit < signal.price

def test_position_size_calculation(strategy):
    entry_price = 100
    stop_loss = 98
    size = strategy._calculate_position_size(entry_price, stop_loss)
    
    # Vérifie que la taille respecte le risque maximum
    risk_amount = size * abs(entry_price - stop_loss)
    max_risk = strategy.position_manager.get_capital() * strategy.parameters.risk_per_trade
    assert abs(risk_amount - max_risk) < 0.01

def test_position_update(strategy, market_data_oversold):
    # Crée une position fictive
    position = strategy.position_manager.create_position(
        direction=1,
        entry_price=90,
        size=1,
        stop_loss=88,
        take_profit=95
    )
    
    # Simule un mouvement de prix vers le target
    strategy._update_positions(
        current_price=93,  # Prix proche du target
        middle_band=95,
        atr=1.0
    )
    
    # Vérifie que le stop a été ajusté
    assert position.stop_loss >= position.entry_price

def test_risk_management(strategy):
    entry_price = 100
    stop_loss = 98
    size = strategy._calculate_position_size(entry_price, stop_loss)
    
    # Vérifie que le risque ne dépasse pas la limite
    max_loss = size * abs(entry_price - stop_loss)
    assert max_loss <= strategy.position_manager.get_capital() * strategy.parameters.risk_per_trade 