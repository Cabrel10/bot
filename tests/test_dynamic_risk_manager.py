"""Tests unitaires pour le module dynamic_risk_manager."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path
import asyncio
import tempfile
import json

from src.core.risk.dynamic_risk_manager import DynamicRiskManager, RiskConfig
from src.core.data_types import MarketData

class TestRiskConfig(unittest.TestCase):
    """Tests pour la classe RiskConfig."""
    
    def setUp(self):
        """Initialise la configuration de test."""
        self.valid_config = RiskConfig(
            max_position_size=100000,
            max_drawdown=0.2
        )
        
    def test_valid_initialization(self):
        """Teste l'initialisation avec des paramètres valides."""
        self.assertEqual(self.valid_config.max_position_size, 100000)
        self.assertEqual(self.valid_config.max_drawdown, 0.2)
        self.assertEqual(self.valid_config.confidence_level, 0.95)
        
    def test_invalid_parameters(self):
        """Teste la validation des paramètres invalides."""
        with self.assertRaises(ValueError):
            RiskConfig(max_position_size=-1000, max_drawdown=0.2)
            
        with self.assertRaises(ValueError):
            RiskConfig(max_position_size=1000, max_drawdown=-0.2)
            
        with self.assertRaises(ValueError):
            config = RiskConfig(max_position_size=1000, max_drawdown=0.2)
            config.confidence_level = 1.5
            
    def test_custom_methods(self):
        """Teste la configuration des méthodes personnalisées."""
        config = RiskConfig(
            max_position_size=1000,
            max_drawdown=0.2,
            position_sizing_method='volatility',
            var_calculation_method='monte_carlo',
            regime_detection_method='threshold'
        )
        
        self.assertEqual(config.position_sizing_method, 'volatility')
        self.assertEqual(config.var_calculation_method, 'monte_carlo')
        self.assertEqual(config.regime_detection_method, 'threshold')
        
class TestDynamicRiskManager(unittest.TestCase):
    """Tests pour la classe DynamicRiskManager."""
    
    def setUp(self):
        """Initialise le gestionnaire de risques et les données de test."""
        self.config = RiskConfig(
            max_position_size=100000,
            max_drawdown=0.2,
            volatility_window=20,
            metrics_history_size=10
        )
        
        self.risk_manager = DynamicRiskManager(self.config)
        
        # Création de données de marché simulées
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        self.market_data = MarketData(
            close=np.random.lognormal(0, 0.1, 100),
            high=np.random.lognormal(0, 0.1, 100) * 1.01,
            low=np.random.lognormal(0, 0.1, 100) * 0.99,
            volume=np.random.lognormal(10, 1, 100),
            timestamp=dates
        )
        
    async def test_update_risk_metrics(self):
        """Teste la mise à jour des métriques de risque."""
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        metrics = self.risk_manager.get_risk_metrics()
        self.assertIn('volatility', metrics)
        self.assertIn('var', metrics)
        self.assertIn('expected_shortfall', metrics)
        
        # Vérifie que les métriques sont dans des plages raisonnables
        self.assertTrue(0 <= metrics['volatility'] <= 1)
        self.assertTrue(-1 <= metrics['var'] <= 0)
        
    async def test_market_regime_detection(self):
        """Teste la détection du régime de marché."""
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        regime = self.risk_manager.get_market_regime()
        self.assertIn(regime, ['bullish', 'bearish', 'neutral'])
        
    async def test_liquidity_analysis(self):
        """Teste l'analyse de la liquidité."""
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        # Vérifie les limites de position basées sur la liquidité
        limits = self.risk_manager.get_current_limits()
        self.assertIn('max_position', limits)
        self.assertIn('max_order_size', limits)
        
        # Vérifie que les limites respectent les contraintes de liquidité
        self.assertTrue(limits['max_position'] <= self.config.max_position_size)
        
    async def test_stress_testing(self):
        """Teste les scénarios de stress test."""
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        results = self.risk_manager.get_stress_test_results()
        self.assertIn('worst_case_loss', results)
        self.assertIn('var_stress', results)
        self.assertIn('es_stress', results)
        
    async def test_alert_generation(self):
        """Teste la génération d'alertes."""
        # Configure des seuils d'alerte bas pour forcer des alertes
        self.risk_manager.config.alert_thresholds = {
            'drawdown': 0.01,
            'var': 0.01,
            'volatility': 0.01
        }
        
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        alerts = self.risk_manager.get_active_alerts()
        self.assertTrue(len(alerts) > 0)
        self.assertIn('timestamp', alerts[0])
        self.assertIn('metric', alerts[0])
        self.assertIn('value', alerts[0])
        
    def test_metrics_history(self):
        """Teste la gestion de l'historique des métriques."""
        async def update_multiple_times():
            for _ in range(15):  # Plus que metrics_history_size
                await self.risk_manager.update_risk_metrics(self.market_data)
                
        asyncio.run(update_multiple_times())
        
        history = self.risk_manager.get_metrics_history()
        
        # Vérifie que l'historique est limité à la taille configurée
        for metric_values in history.values():
            self.assertLessEqual(len(metric_values), self.config.metrics_history_size)
            
    async def test_state_persistence(self):
        """Teste la sauvegarde et le chargement de l'état."""
        await self.risk_manager.update_risk_metrics(self.market_data)
        
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            # Sauvegarde l'état
            self.risk_manager.save_state(tmp.name)
            
            # Crée un nouveau gestionnaire
            new_manager = DynamicRiskManager(self.config)
            new_manager.load_state(tmp.name)
            
            # Vérifie que l'état est correctement restauré
            self.assertEqual(
                self.risk_manager.get_risk_metrics(),
                new_manager.get_risk_metrics()
            )
            self.assertEqual(
                self.risk_manager.get_market_regime(),
                new_manager.get_market_regime()
            )
            
    def test_error_handling(self):
        """Teste la gestion des erreurs."""
        async def test_invalid_data():
            # Teste avec des données invalides
            invalid_data = MarketData(
                close=np.array([]),
                high=np.array([]),
                low=np.array([]),
                volume=np.array([]),
                timestamp=pd.DatetimeIndex([])
            )
            
            with self.assertRaises(ValueError):
                await self.risk_manager.update_risk_metrics(invalid_data)
                
            # Teste avec des données contenant des NaN
            nan_data = MarketData(
                close=np.array([np.nan, 1, 2]),
                high=np.array([1, 2, 3]),
                low=np.array([0, 1, 2]),
                volume=np.array([100, 200, 300]),
                timestamp=pd.date_range(start='2024-01-01', periods=3)
            )
            
            with self.assertRaises(ValueError):
                await self.risk_manager.update_risk_metrics(nan_data)
                
        asyncio.run(test_invalid_data())

if __name__ == '__main__':
    unittest.main() 