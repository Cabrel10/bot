"""
Configuration globale des tests.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Ajout du chemin du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import des modules du projet
from trading.models.hybrid_model.model import HybridModel
from trading.services.backtesting.realtime_backtester import RealtimeBacktester, RealtimeConfig
from trading.services.backtesting.risk_metrics import RiskAnalyzer, RiskMetrics
from trading.core.exchanges.base import BaseExchange

@pytest.fixture(scope="session")
def project_root():
    """Retourne le chemin racine du projet."""
    return project_root

@pytest.fixture(scope="session")
def sample_data():
    """Crée des données de test communes."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
    return {
        "dates": dates,
        "prices": pd.Series(np.random.normal(50000, 1000, 100), index=dates),
        "volumes": pd.Series(np.random.normal(100, 20, 100), index=dates)
    }

@pytest.fixture(scope="session")
def mock_exchange():
    """Crée un mock de l'exchange."""
    exchange = pytest.Mock(spec=BaseExchange)
    exchange.get_websocket_data = pytest.AsyncMock()
    exchange.get_klines = pytest.AsyncMock()
    return exchange

@pytest.fixture(scope="session")
def mock_model():
    """Crée un mock du modèle hybride."""
    model = pytest.Mock(spec=HybridModel)
    model.predict = pytest.Mock(return_value={"direction": 1.0})
    return model

@pytest.fixture(scope="session")
def backtest_config():
    """Crée une configuration de backtest."""
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

@pytest.fixture(scope="session")
def sample_trades():
    """Crée des trades de test."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": ["BTC/USDT"] * 10,
        "pnl": [100, -50, 150, -30, 80, 40, -100, 60, -20, 30],
        "return": [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, -0.01, 0.01]
    })

@pytest.fixture(scope="session")
def sample_returns():
    """Crée des rendements de test."""
    return pd.Series([
        0.01, -0.02, 0.03, -0.01, 0.02,
        0.01, -0.03, 0.02, -0.01, 0.01
    ])

@pytest.fixture(scope="session")
def risk_analyzer(sample_returns, sample_trades):
    """Crée une instance de RiskAnalyzer."""
    return RiskAnalyzer(sample_returns, sample_trades)

@pytest.fixture(scope="session")
def backtester(backtest_config, mock_exchange):
    """Crée une instance de RealtimeBacktester."""
    return RealtimeBacktester(backtest_config, mock_exchange)

@pytest.fixture(scope="function")
def event_loop():
    """Crée une nouvelle boucle d'événements pour chaque test."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_logging():
    """Configure le logging pour les tests."""
    import logging
    logging.basicConfig(level=logging.INFO)
    yield

@pytest.fixture(autouse=True)
def disable_tensorflow_logging():
    """Désactive les logs TensorFlow pendant les tests."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    yield