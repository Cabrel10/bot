"""
Tests for core components.
"""

import pytest
from datetime import datetime
from trading.core.data_types import (
    OrderType,
    OrderSide,
    TimeFrame,
    MarketData,
    OrderData,
    TradeData,
    PositionData,
    BaseData
)
from trading.core.exchanges import BaseExchange
from trading.core.strategies import BaseStrategy
from trading.core.risk import RiskManager
from trading.core.position import PositionManager

@pytest.fixture
def market_data():
    """Create sample market data."""
    return MarketData(
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        timeframe="1h"
    )

@pytest.fixture
def order_data():
    """Create sample order data."""
    return OrderData(
        order_id="test_order",
        symbol="BTC/USDT",
        order_type="market",
        side="buy",
        quantity=1.0,
        price=50000.0,
        timestamp=datetime.now()
    )

@pytest.fixture
def position_data():
    """Create sample position data."""
    return PositionData(
        symbol="BTC/USDT",
        quantity=1.0,
        average_price=50000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        timestamp=datetime.now()
    )

@pytest.fixture
def risk_manager():
    """Create risk manager instance."""
    return RiskManager(
        max_position_size=1.0,
        max_drawdown=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_leverage=1.0
    )

@pytest.fixture
def position_manager():
    """Create position manager instance."""
    return PositionManager(initial_capital=100000.0)

def test_base_data():
    """Test base data functionality."""
    data = BaseData()
    assert isinstance(data.metadata, dict)
    data.add_metadata("test", "value")
    assert data.get_metadata("test") == "value"

def test_market_data(market_data):
    """Test market data creation and properties."""
    assert isinstance(market_data, BaseData)
    assert market_data.symbol == "BTC/USDT"
    assert market_data.open == 50000.0
    assert market_data.high == 51000.0
    assert market_data.low == 49000.0
    assert market_data.close == 50500.0
    assert market_data.volume == 100.0
    assert market_data.timeframe == "1h"

def test_order_data(order_data):
    """Test order data creation and properties."""
    assert isinstance(order_data, BaseData)
    assert order_data.order_id == "test_order"
    assert order_data.symbol == "BTC/USDT"
    assert order_data.order_type == "market"
    assert order_data.side == "buy"
    assert order_data.quantity == 1.0
    assert order_data.price == 50000.0

def test_position_data(position_data):
    """Test position data creation and properties."""
    assert isinstance(position_data, BaseData)
    assert position_data.symbol == "BTC/USDT"
    assert position_data.quantity == 1.0
    assert position_data.average_price == 50000.0
    assert position_data.unrealized_pnl == 0.0
    assert position_data.realized_pnl == 0.0

def test_risk_manager(risk_manager, market_data, position_data):
    """Test risk manager functionality."""
    # Test position size calculation
    position_size = risk_manager.calculate_position_size(
        capital=100000.0,
        price=50000.0,
        volatility=0.02
    )
    assert position_size > 0
    assert position_size <= risk_manager.max_position_size
    
    # Test risk metrics
    risk_metrics = risk_manager.update_position_risk(
        position_data,
        market_data.close
    )
    assert 'unrealized_pnl' in risk_metrics
    assert 'stop_loss' in risk_metrics
    assert 'take_profit' in risk_metrics
    
    # Test risk limits
    assert risk_manager.check_risk_limits(position_data, market_data)

def test_position_manager(position_manager, order_data, position_data):
    """Test position manager functionality."""
    # Test opening position
    position = position_manager.open_position(
        symbol="BTC/USDT",
        quantity=1.0,
        price=50000.0,
        side="buy"
    )
    assert position.symbol == "BTC/USDT"
    assert position.quantity == 1.0
    assert position.average_price == 50000.0
    
    # Test position update
    position_manager.update_position("BTC/USDT", 51000.0)
    position = position_manager.get_position("BTC/USDT")
    assert position.unrealized_pnl == 1000.0
    
    # Test closing position
    trade = position_manager.close_position("BTC/USDT", 51000.0)
    assert trade is not None
    assert trade.symbol == "BTC/USDT"
    assert trade.quantity == 1.0
    assert trade.price == 51000.0
    
    # Test order management
    position_manager.add_order(order_data)
    assert position_manager.get_order("test_order") is not None
    position_manager.remove_order("test_order")
    assert position_manager.get_order("test_order") is None 