import unittest
import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import warnings
import logging
import sys
import os
import pytest_asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.hybrid_model.integration import (
    CryptoTradingSystem,
    CryptoMetaTrader,
    CryptoGAN,
    HybridModel,
    MarketRegimeManager,
    RiskManager
)

# Configuration TensorFlow pour optimisations CPU
tf.config.optimizer.set_jit(True)  # Enable XLA
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Filtrer les avertissements spécifiques
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

@pytest.mark.asyncio
class TestCryptoTradingSystem:
    @pytest_asyncio.fixture(autouse=True)
    async def setup_system(self):
        """Configuration initiale pour tous les tests."""
        self.config = {
            'meta_learning': {
                'base_lr': 0.001,
                'adaptation_steps': 3,
                'features': ['close', 'high', 'low', 'open', 'volume']
            },
            'neat': {
                'population_size': 20,
                'generations': 5
            },
            'gan': {
                'latent_dim': 100,
                'generator_dim': 128,
                'discriminator_dim': 128,
                'learning_rate': 0.0002,
                'beta1': 0.5,
                'n_features': 5
            },
            'features': ['close', 'high', 'low', 'open', 'volume'],
            'timeframes': ['1h'],
            'batch_size': 32,
            'risk': {
                'base_risk_factor': 0.8,
                'market_risk_weight': 0.4,
                'position_risk_weight': 0.3,
                'volatility_risk_weight': 0.3
            },
            'add_technical_features': True,
            'use_synthetic': True,
            'synthetic_samples': 100,
            'evolve_on_regime_change': True,
            'evolution_generations': 5,
            'adaptation_steps': 10,
            'adaptation_lr': 0.001
        }
        self.system = CryptoTradingSystem(self.config)
        yield
        # Nettoyage après les tests
        await self.system.cleanup()

    def generate_mock_data(self):
        """Génère des données de test réalistes."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1h'
        )
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.normal(35000, 1000, len(dates)),
            'volume': np.random.normal(1000000, 100000, len(dates)),
            'high': np.random.normal(35500, 1000, len(dates)),
            'low': np.random.normal(34500, 1000, len(dates)),
            'open': np.random.normal(35000, 1000, len(dates))
        })
        
        return data

    @pytest.mark.asyncio
    async def test_market_update_processing(self):
        """Test du traitement des mises à jour du marché."""
        mock_data = self.generate_mock_data()
        result = await self.system.process_market_update(mock_data.iloc[0].to_dict())
        
        assert isinstance(result, dict)
        assert 'signal' in result
        assert 'confidence' in result
        assert 'regime' in result
        assert 'risk_score' in result
        
        assert -1 <= result['signal'] <= 1
        assert 0 <= result['confidence'] <= 1
        assert result['regime'] in ['volatile', 'trending', 'ranging']

    @pytest.mark.asyncio
    async def test_regime_detection(self):
        """Test de la détection des régimes de marché."""
        mock_data = self.generate_mock_data()
        regime = self.system.regime_manager.detect_regime(mock_data.iloc[-20:].to_dict())
        assert regime in ['volatile', 'trending', 'ranging']

    @pytest.mark.asyncio
    async def test_risk_management(self):
        """Test de la gestion des risques."""
        raw_signal = 0.8
        adjusted_signal = self.system.risk_manager.adjust_signal(raw_signal)
        assert abs(adjusted_signal) <= abs(raw_signal)
        assert -1 <= adjusted_signal <= 1

    @pytest.mark.asyncio
    async def test_synthetic_data_generation(self):
        """Test de la génération de données synthétiques."""
        mock_data = self.generate_mock_data()
        synthetic_data = await self.system._generate_synthetic_data(mock_data)
        assert isinstance(synthetic_data, pd.DataFrame)
        assert not synthetic_data.empty
        for feature in self.config['features']:
            assert feature in synthetic_data.columns

    @pytest.mark.asyncio
    async def test_model_components(self):
        """Test des composants individuels du modèle."""
        assert isinstance(self.system.meta_trader, CryptoMetaTrader)
        assert isinstance(self.system.gan, CryptoGAN)
        assert self.system.config['batch_size'] == self.config['batch_size']

    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test des métriques de performance."""
        mock_data = self.generate_mock_data()
        signals = []
        for i in range(len(mock_data) - 1):
            result = await self.system.process_market_update(mock_data.iloc[i].to_dict())
            signals.append(result['signal'])
        
        assert len(signals) > 0
        assert all(isinstance(s, float) for s in signals)

@pytest.mark.asyncio
class TestHybridModel:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self):
        """Configuration initiale pour chaque test."""
        self.config = {
            'meta_learning': {
                'base_lr': 0.001,
                'adaptation_steps': 3
            },
            'neat': {
                'population_size': 20,
                'generations': 5
            },
            'gan': {
                'latent_dim': 100,
                'generator_dim': 128,
                'discriminator_dim': 128,
                'learning_rate': 0.0002,
                'beta1': 0.5
            },
            'features': ['close', 'volume', 'high', 'low', 'open'],
            'timeframes': ['1h'],
            'batch_size': 32,
            'risk': {
                'base_risk_factor': 0.8,
                'market_risk_weight': 0.4,
                'position_risk_weight': 0.3,
                'volatility_risk_weight': 0.3
            },
            'add_technical_features': True
        }
        self.model = HybridModel(self.config)
        self.test_data = self.generate_test_data()
        yield
        # Nettoyage
        await self.model.cleanup()

    def generate_test_data(self):
        """Génère des données de test."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1h'
        )
        
        n = len(dates)
        regime1 = np.random.normal(0, 1, n//3)
        regime2 = np.random.normal(2, 1.5, n//3)
        regime3 = np.random.normal(-1, 0.5, n - 2*(n//3))
        price = np.concatenate([regime1, regime2, regime3])
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': price,
            'volume': np.random.normal(1000000, 100000, n),
            'high': price * 1.01,
            'low': price * 0.99,
            'open': price
        })

    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Teste l'initialisation du modèle."""
        assert self.model is not None

    @pytest.mark.asyncio
    async def test_regime_detection(self):
        """Teste la détection des changements de régime."""
        change_points = self.model.detect_regime_changes(self.test_data)
        assert isinstance(change_points, list)
        assert len(change_points) > 0

    @pytest.mark.asyncio
    async def test_data_preprocessing(self):
        """Teste le prétraitement des données."""
        processed_data = self.model.preprocess_data(self.test_data)
        assert processed_data is not None
        assert isinstance(processed_data, np.ndarray)
        assert len(processed_data.shape) == 2  # S'assure que c'est une matrice 2D

class TestBasicFunctionality:
    """Tests de base sans dépendance à TensorFlow."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Configuration initiale pour chaque test."""
        self.test_data = self.generate_test_data()

    def generate_test_data(self):
        """Génère des données de test."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='h'
        )
        
        n = len(dates)
        trend = np.linspace(0, 1, n)
        noise = np.random.normal(0, 0.1, n)
        price = 1000 * (1 + trend + noise)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': price,
            'volume': np.random.normal(1000000, 100000, n),
            'high': price * 1.01,
            'low': price * 0.99,
            'open': price
        })

    def test_data_generation(self):
        """Teste la génération des données."""
        assert self.test_data is not None
        assert len(self.test_data) > 0
        required_columns = ['timestamp', 'close', 'volume', 'high', 'low', 'open']
        assert all(col in self.test_data.columns for col in required_columns)

    def test_data_integrity(self):
        """Teste l'intégrité des données."""
        assert all(self.test_data['high'] >= self.test_data['low'])
        assert all(self.test_data['volume'] > 0)

if __name__ == '__main__':
    pytest.main([__file__, "-v"]) 