import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.strategies.base_strategy import BaseStrategy
from src.core.data_types import MarketData, Signal, Position
from src.core.exceptions import InvalidDataError, InsufficientDataError
from src.core.position import PositionManager
from src.core.risk import RiskManager

class TestStrategy(BaseStrategy):
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        signals = []
        if len(market_data.close) > 1:
            if market_data.close[-1] > market_data.close[-2]:
                signals.append(Signal(
                    timestamp=market_data.timestamp[-1],
                    direction=1,
                    strength=1.0
                ))
        return signals

@pytest.fixture
def market_data():
    n_points = 200
    timestamps = [
        datetime.now() + timedelta(minutes=i) 
        for i in range(n_points)
    ]
    close_prices = np.random.random(n_points) + 100
    
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
    position_manager = PositionManager()
    risk_manager = RiskManager()
    return TestStrategy(position_manager, risk_manager)

def test_strategy_initialization(strategy):
    assert strategy.name == "TestStrategy"
    assert isinstance(strategy.position_manager, PositionManager)
    assert isinstance(strategy.risk_manager, RiskManager)

def test_update(strategy, market_data):
    signals, positions = strategy.update(market_data)
    assert isinstance(signals, list)
    assert isinstance(positions, list)
    assert strategy.market_state.get('current_price') is not None

def test_market_state_update(strategy, market_data):
    strategy._update_market_state(market_data)
    assert 'current_price' in strategy.market_state
    assert 'volatility' in strategy.market_state
    assert 'trend' in strategy.market_state

def test_invalid_data(strategy):
    invalid_data = MarketData(
        timestamp=[datetime.now()],
        open=[100],
        high=None,  # Invalid
        low=[99],
        close=[100],
        volume=[1000]
    )
    with pytest.raises(InvalidDataError):
        strategy.validate_data(invalid_data)

def test_metrics_update(strategy):
    positions = [
        Position(entry_price=100, exit_price=110, size=1),
        Position(entry_price=110, exit_price=105, size=1)
    ]
    strategy._update_metrics(positions)
    assert strategy.current_metrics.number_of_trades == 2
    assert strategy.current_metrics.total_return != 0

def test_get_state(strategy, market_data):
    strategy.update(market_data)
    state = strategy.get_state()
    assert 'name' in state
    assert 'market_state' in state
    assert 'current_metrics' in state
    assert 'active_positions' in state 