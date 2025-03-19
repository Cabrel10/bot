"""
Integration tests for the trading system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from trading.core.data_types import (
    MarketData,
    OrderData,
    PositionData,
    BaseData
)
from trading.services.execution.execution_engine import ExecutionEngine
from trading.services.execution.order_manager import OrderManager
from trading.services.execution.position_manager import PositionManager

@pytest.fixture
def execution_engine():
    """Create execution engine instance."""
    return ExecutionEngine(initial_capital=100000.0)

@pytest.fixture
def market_data():
    """Create sample market data."""
    return {
        "BTC/USDT": 50000.0,
        "ETH/USDT": 3000.0,
        "BNB/USDT": 400.0
    }

@pytest.fixture
def sample_orders():
    """Create sample orders."""
    return [
        {
            "symbol": "BTC/USDT",
            "order_type": "market",
            "side": "buy",
            "quantity": 0.1,
            "price": None
        },
        {
            "symbol": "ETH/USDT",
            "order_type": "limit",
            "side": "sell",
            "quantity": 1.0,
            "price": 3100.0
        }
    ]

@pytest.mark.asyncio
async def test_order_execution(execution_engine, sample_orders):
    """Test order execution flow."""
    # Execute orders
    results = []
    for order in sample_orders:
        result = execution_engine.execute_order(**order)
        if result:
            results.append(result)
    
    # Verify results
    assert len(results) == len(sample_orders)
    for result, order in zip(results, sample_orders):
        assert result.symbol == order["symbol"]
        assert result.order_type == order["order_type"]
        assert result.side == order["side"]
        assert result.quantity == order["quantity"]
        assert result.status == "filled"

@pytest.mark.asyncio
async def test_position_management(execution_engine, market_data):
    """Test position management flow."""
    # Open positions
    positions = []
    for symbol, price in market_data.items():
        result = execution_engine.execute_order(
            symbol=symbol,
            order_type="market",
            side="buy",
            quantity=0.1,
            price=None
        )
        if result and result.position_id:
            positions.append(result)
    
    # Verify positions
    assert len(positions) == len(market_data)
    for position in positions:
        assert position.position_id in market_data
        assert position.position_quantity > 0
        assert position.position_average_price > 0
    
    # Update positions
    updated_market_data = {
        symbol: price * 1.1  # 10% price increase
        for symbol, price in market_data.items()
    }
    results = execution_engine.update_positions(updated_market_data)
    
    # Verify updates
    assert len(results) == 0  # No liquidations
    summary = execution_engine.get_position_summary()
    assert summary["active_positions"] == len(market_data)
    assert summary["total_pnl"] > 0  # Should have profit

@pytest.mark.asyncio
async def test_stop_loss_execution(execution_engine, market_data):
    """Test stop loss execution."""
    # Open position
    symbol = "BTC/USDT"
    result = execution_engine.execute_order(
        symbol=symbol,
        order_type="market",
        side="buy",
        quantity=0.1,
        price=None
    )
    
    assert result and result.position_id
    
    # Set stop loss
    position = execution_engine._position_manager.get_position(symbol)
    assert position is not None
    execution_engine._position_manager.set_stop_loss(
        symbol=symbol,
        price=market_data[symbol] * 0.95  # 5% below entry
    )
    
    # Update with price below stop loss
    updated_market_data = {
        symbol: market_data[symbol] * 0.94  # 6% below entry
    }
    results = execution_engine.update_positions(updated_market_data)
    
    # Verify liquidation
    assert len(results) == 1
    assert results[0].position_id == symbol
    assert results[0].position_realized_pnl < 0  # Should have loss

@pytest.mark.asyncio
async def test_take_profit_execution(execution_engine, market_data):
    """Test take profit execution."""
    # Open position
    symbol = "ETH/USDT"
    result = execution_engine.execute_order(
        symbol=symbol,
        order_type="market",
        side="buy",
        quantity=1.0,
        price=None
    )
    
    assert result and result.position_id
    
    # Set take profit
    position = execution_engine._position_manager.get_position(symbol)
    assert position is not None
    execution_engine._position_manager.set_take_profit(
        symbol=symbol,
        price=market_data[symbol] * 1.05  # 5% above entry
    )
    
    # Update with price above take profit
    updated_market_data = {
        symbol: market_data[symbol] * 1.06  # 6% above entry
    }
    results = execution_engine.update_positions(updated_market_data)
    
    # Verify liquidation
    assert len(results) == 1
    assert results[0].position_id == symbol
    assert results[0].position_realized_pnl > 0  # Should have profit

@pytest.mark.asyncio
async def test_order_history(execution_engine, sample_orders):
    """Test order history tracking."""
    # Execute orders
    for order in sample_orders:
        execution_engine.execute_order(**order)
    
    # Get history
    history = execution_engine.get_execution_history()
    assert len(history) == len(sample_orders)
    
    # Verify order details
    for result, order in zip(history, sample_orders):
        assert result.symbol == order["symbol"]
        assert result.order_type == order["order_type"]
        assert result.side == order["side"]
        assert result.quantity == order["quantity"]
        assert result.status == "filled"

@pytest.mark.asyncio
async def test_performance_metrics(execution_engine, market_data):
    """Test performance metrics calculation."""
    # Open and close positions
    for symbol, price in market_data.items():
        # Open position
        execution_engine.execute_order(
            symbol=symbol,
            order_type="market",
            side="buy",
            quantity=0.1,
            price=None
        )
        
        # Update with profit
        execution_engine.update_positions({
            symbol: price * 1.1  # 10% profit
        })
        
        # Close position
        execution_engine.update_positions({
            symbol: price * 1.1  # Keep profit
        })
    
    # Get summaries
    position_summary = execution_engine.get_position_summary()
    order_summary = execution_engine.get_order_summary()
    
    # Verify metrics
    assert position_summary["total_positions"] == len(market_data)
    assert position_summary["total_trades"] == len(market_data) * 2  # Open and close
    assert position_summary["total_pnl"] > 0  # Should have profit
    assert order_summary["total_orders"] == len(market_data) * 2  # Open and close orders
    assert order_summary["filled_orders"] == len(market_data) * 2 