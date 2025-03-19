"""
Tests unitaires pour la stratégie de croisement de moyennes mobiles.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.plugins.strategies.ma_cross import (
    MACrossStrategy,
    MarketType,
    PositionConfig,
    TrendAnalysis,
    SignalError,
    PositionError
)

class TestMACrossStrategy(unittest.TestCase):
    """Tests pour la classe MACrossStrategy."""
    
    def setUp(self):
        """Initialisation des données de test."""
        # Création de données OHLCV de test
        dates = pd.date_range(
            start='2024-01-01',
            end='2024-01-30',
            freq='1H'
        )
        
        # Création d'une tendance artificielle
        trend = np.concatenate([
            np.linspace(100, 120, len(dates)//3),
            np.linspace(120, 90, len(dates)//3),
            np.linspace(90, 110, len(dates)//3 + len(dates)%3)
        ])
        
        noise = np.random.normal(0, 2, len(dates))
        price = trend + noise
        
        self.test_data = pd.DataFrame(
            {
                'open': price - 1,
                'high': price + 1,
                'low': price - 1,
                'close': price,
                'volume': np.random.normal(1000, 100, len(dates))
            },
            index=dates
        )
        
        # Données de funding rate pour les futures
        self.funding_data = pd.Series(
            np.random.normal(0.001, 0.0002, len(dates)),
            index=dates
        )
        
        # Configuration de base pour les tests
        self.position_config = PositionConfig(
            max_position_size=1.0,
            risk_per_trade=0.02,
            max_leverage=1.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015
        )
        
        # Instance de base pour les tests
        self.strategy = MACrossStrategy(
            position_config=self.position_config
        )
        
    def test_init_validation(self):
        """Test de la validation des paramètres d'initialisation."""
        # Test des paramètres invalides
        with self.assertRaises(ValueError):
            MACrossStrategy(short_window=0)
            
        with self.assertRaises(ValueError):
            MACrossStrategy(
                short_window=30,
                long_window=10
            )
            
        with self.assertRaises(ValueError):
            MACrossStrategy(min_trend_strength=-0.1)
            
        # Test des paramètres valides
        try:
            MACrossStrategy(
                short_window=10,
                long_window=30,
                min_trend_strength=0.02,
                volume_factor=1.5
            )
        except ValueError:
            self.fail("L'initialisation avec des paramètres valides a échoué")
            
    def test_data_validation(self):
        """Test de la validation des données d'entrée."""
        # Test avec données manquantes
        with self.assertRaises(SignalError):
            self.strategy.analyze({})
            
        # Test avec colonnes manquantes
        invalid_data = {'ohlcv': pd.DataFrame({'close': [1, 2, 3]})}
        with self.assertRaises(SignalError):
            self.strategy.analyze(invalid_data)
            
        # Test avec données futures sans funding rate
        futures_strategy = MACrossStrategy(
            market_type=MarketType.FUTURES
        )
        with self.assertRaises(SignalError):
            futures_strategy.analyze({'ohlcv': self.test_data})
            
    def test_trend_analysis(self):
        """Test de l'analyse de tendance."""
        data = {'ohlcv': self.test_data}
        result = self.strategy.analyze(data)
        
        # Vérification de la structure de l'analyse
        trend_data = result['meta']['trend']
        self.assertIn('direction', trend_data)
        self.assertIn('strength', trend_data)
        self.assertIn('duration', trend_data)
        self.assertIn('volatility', trend_data)
        self.assertIn('volume_support', trend_data)
        
        # Test des valeurs
        self.assertIn(trend_data['direction'], [-1, 0, 1])
        self.assertTrue(0 <= trend_data['strength'] <= 1)
        self.assertTrue(trend_data['duration'] >= 0)
        self.assertTrue(trend_data['volatility'] >= 0)
        self.assertIsInstance(trend_data['volume_support'], bool)
        
    def test_signal_generation(self):
        """Test de la génération des signaux."""
        data = {'ohlcv': self.test_data}
        result = self.strategy.analyze(data)
        
        # Vérification des signaux
        self.assertIn('action', result)
        self.assertIn('position_size', result)
        self.assertIn('risk_levels', result)
        
        # Test des valeurs
        if result['action'] is not None:
            self.assertIn(result['action'], ['buy', 'sell'])
            self.assertTrue(0 <= result['position_size'] <= 1)
            
        # Test des niveaux de risque
        risk_levels = result['risk_levels']
        if result['action'] is not None:
            self.assertIsNotNone(risk_levels['stop_loss'])
            self.assertIsNotNone(risk_levels['take_profit'])
            
    def test_futures_analysis(self):
        """Test de l'analyse pour les futures."""
        futures_strategy = MACrossStrategy(
            market_type=MarketType.FUTURES,
            position_config=self.position_config
        )
        
        data = {
            'ohlcv': self.test_data,
            'funding_rate': self.funding_data
        }
        
        result = futures_strategy.analyze(data)
        
        # Vérification des données futures
        self.assertIn('futures_data', result['meta'])
        futures_data = result['meta']['futures_data']
        if futures_data is not None:
            self.assertIn('funding_rate', futures_data)
            self.assertIn('funding_ma', futures_data)
            self.assertIn('funding_signal', futures_data)
            
    def test_position_sizing(self):
        """Test du calcul de la taille des positions."""
        data = {'ohlcv': self.test_data}
        result = self.strategy.analyze(data)
        
        # Test des limites de taille
        self.assertTrue(
            0 <= result['position_size'] <= self.position_config.max_position_size
        )
        
        # Test avec tendance forte
        strong_trend_data = self.test_data.copy()
        strong_trend_data['close'] = np.linspace(100, 200, len(self.test_data))
        result_strong = self.strategy.analyze(
            {'ohlcv': strong_trend_data}
        )
        
        # La taille devrait être plus grande avec une tendance forte
        if (
            result['action'] is not None
            and result_strong['action'] is not None
        ):
            self.assertGreater(
                result_strong['position_size'],
                result['position_size']
            )
            
    def test_risk_levels(self):
        """Test du calcul des niveaux de risque."""
        data = {'ohlcv': self.test_data}
        result = self.strategy.analyze(data)
        
        if result['action'] is not None:
            risk_levels = result['risk_levels']
            current_price = self.test_data['close'].iloc[-1]
            
            # Test du stop loss
            self.assertLess(
                risk_levels['stop_loss'],
                current_price
            )
            
            # Test du take profit
            self.assertGreater(
                risk_levels['take_profit'],
                current_price
            )
            
            # Test du trailing stop
            if risk_levels['trailing_stop'] is not None:
                self.assertLess(
                    risk_levels['trailing_stop'],
                    current_price
                )
                
    def test_error_handling(self):
        """Test de la gestion des erreurs."""
        # Test avec données invalides
        with self.assertRaises(SignalError):
            self.strategy.analyze({
                'ohlcv': pd.DataFrame({'close': [np.nan, np.inf, -np.inf]})
            })
            
        # Test avec données vides
        with self.assertRaises(SignalError):
            self.strategy.analyze({
                'ohlcv': pd.DataFrame()
            })
            
if __name__ == '__main__':
    unittest.main() 