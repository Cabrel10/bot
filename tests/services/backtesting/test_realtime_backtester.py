"""
Tests unitaires pour le RealtimeBacktester.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from src.services.backtesting.realtime_backtester import (
    RealtimeBacktester,
    RealtimeConfig
)
from src.core.data_types import OrderType, OrderSide, TimeFrame
from src.models.hybrid_model.model import HybridModel

@pytest.fixture
def mock_exchange():
    """Crée un mock de l'exchange."""
    exchange = Mock()
    exchange.get_websocket_data = AsyncMock()
    exchange.get_klines = AsyncMock()
    return exchange

@pytest.fixture
def mock_model():
    """Crée un mock du modèle hybride."""
    model = Mock(spec=HybridModel)
    model.predict = Mock(return_value={"direction": 1.0})
    return model

@pytest.fixture
def config():
    """Crée une configuration de test."""
    return RealtimeConfig(
        symbols=["BTC/USDT", "ETH/USDT"],
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now(),
        initial_balance=10000.0,
        update_interval=1,
        buffer_size=100,
        max_latency=0.5,
        use_websocket=True
    )

@pytest.fixture
def backtester(config, mock_exchange):
    """Crée une instance du backtester."""
    return RealtimeBacktester(config, mock_exchange)

@pytest.mark.asyncio
async def test_initialization(backtester, config):
    """Teste l'initialisation du backtester."""
    assert backtester.config == config
    assert backtester.is_running is False
    assert len(backtester.data_buffer) == len(config.symbols)
    assert len(backtester.last_update) == len(config.symbols)

@pytest.mark.asyncio
async def test_start_stop(backtester, mock_model):
    """Teste le démarrage et l'arrêt du backtester."""
    # Démarrage
    start_task = asyncio.create_task(backtester.start(mock_model))
    assert backtester.is_running is True
    
    # Arrêt après un court délai
    await asyncio.sleep(0.1)
    backtester.stop()
    assert backtester.is_running is False
    
    # Nettoyage
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_market_data_update(backtester, mock_exchange):
    """Teste la mise à jour des données de marché."""
    # Configuration des données mockées
    mock_data = {
        "timestamp": datetime.now(),
        "open": 50000.0,
        "high": 51000.0,
        "low": 49000.0,
        "close": 50500.0,
        "volume": 100.0
    }
    mock_exchange.get_websocket_data.return_value = mock_data
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(Mock()))
    
    # Attente de la mise à jour
    await asyncio.sleep(0.1)
    
    # Vérification
    assert len(backtester.data_buffer["BTC/USDT"]) > 0
    assert backtester.data_buffer["BTC/USDT"][-1] == mock_data
    
    # Nettoyage
    backtester.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_signal_processing(backtester, mock_model):
    """Teste le traitement des signaux."""
    # Configuration des données mockées
    mock_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [50000.0],
        "high": [51000.0],
        "low": [49000.0],
        "close": [50500.0],
        "volume": [100.0]
    })
    backtester.data_buffer["BTC/USDT"] = [mock_data.iloc[0].to_dict()]
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(mock_model))
    
    # Attente du traitement
    await asyncio.sleep(0.1)
    
    # Vérification
    assert mock_model.predict.called
    
    # Nettoyage
    backtester.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_order_execution(backtester, mock_model):
    """Teste l'exécution des ordres."""
    # Configuration des données mockées
    mock_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [50000.0],
        "high": [51000.0],
        "low": [49000.0],
        "close": [50500.0],
        "volume": [100.0]
    })
    backtester.data_buffer["BTC/USDT"] = [mock_data.iloc[0].to_dict()]
    
    # Configuration du signal
    mock_model.predict.return_value = {
        "direction": 1.0,
        "confidence": 0.8
    }
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(mock_model))
    
    # Attente de l'exécution
    await asyncio.sleep(0.1)
    
    # Vérification
    assert len(backtester.orders) > 0
    
    # Nettoyage
    backtester.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_performance_monitoring(backtester, mock_model):
    """Teste le monitoring des performances."""
    # Configuration des données mockées
    mock_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [50000.0],
        "high": [51000.0],
        "low": [49000.0],
        "close": [50500.0],
        "volume": [100.0]
    })
    backtester.data_buffer["BTC/USDT"] = [mock_data.iloc[0].to_dict()]
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(mock_model))
    
    # Attente du monitoring
    await asyncio.sleep(0.1)
    
    # Vérification
    metrics = backtester._calculate_realtime_metrics()
    assert "equity" in metrics
    assert "open_positions" in metrics
    assert "total_trades" in metrics
    
    # Nettoyage
    backtester.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_stop_conditions(backtester, mock_model):
    """Teste les conditions d'arrêt."""
    # Configuration des données mockées
    mock_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "open": [50000.0],
        "high": [51000.0],
        "low": [49000.0],
        "close": [50500.0],
        "volume": [100.0]
    })
    backtester.data_buffer["BTC/USDT"] = [mock_data.iloc[0].to_dict()]
    
    # Simulation d'un drawdown important
    backtester.equity_curve = [
        {"timestamp": datetime.now(), "equity": 10000.0},
        {"timestamp": datetime.now(), "equity": 8000.0},
        {"timestamp": datetime.now(), "equity": 4000.0}
    ]
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(mock_model))
    
    # Attente de la vérification des conditions
    await asyncio.sleep(0.1)
    
    # Vérification
    assert not backtester.is_running
    
    # Nettoyage
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_error_handling(backtester, mock_model, mock_exchange):
    """Teste la gestion des erreurs."""
    # Simulation d'une erreur de l'exchange
    mock_exchange.get_websocket_data.side_effect = Exception("Erreur de connexion")
    
    # Démarrage du backtester
    start_task = asyncio.create_task(backtester.start(mock_model))
    
    # Attente de la gestion de l'erreur
    await asyncio.sleep(0.1)
    
    # Vérification que le backtester continue de fonctionner
    assert backtester.is_running
    
    # Nettoyage
    backtester.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass 