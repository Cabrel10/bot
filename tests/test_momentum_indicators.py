"""
Tests unitaires pour les indicateurs de momentum.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.plugins.indicators.momentum import (
    MomentumIndicators,
    MarketType,
    VolumeProfile,
    IndicatorError,
    DataValidationError,
    CalculationError
)

class TestMomentumIndicators(unittest.TestCase):
    """Tests pour la classe MomentumIndicators."""
    
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
        
        # Instance de base pour les tests
        self.indicators = MomentumIndicators()
        
    def test_init_validation(self):
        """Test de la validation des paramètres d'initialisation."""
        # Test des paramètres invalides
        with self.assertRaises(ValueError):
            MomentumIndicators(rsi_window=0)
            
        with self.assertRaises(ValueError):
            MomentumIndicators(rsi_overbought=120)
            
        with self.assertRaises(ValueError):
            MomentumIndicators(rsi_oversold=-10)
            
        with self.assertRaises(ValueError):
            MomentumIndicators(
                rsi_overbought=50,
                rsi_oversold=60
            )
            
        # Test des paramètres valides
        try:
            MomentumIndicators(
                rsi_window=14,
                rsi_overbought=70,
                rsi_oversold=30,
                volume_ma_window=20
            )
        except ValueError:
            self.fail("L'initialisation avec des paramètres valides a échoué")
            
    def test_data_validation(self):
        """Test de la validation des données d'entrée."""
        # Test avec données manquantes
        with self.assertRaises(DataValidationError):
            self.indicators.process({})
            
        # Test avec colonnes manquantes
        invalid_data = {'ohlcv': pd.DataFrame({'close': [1, 2, 3]})}
        with self.assertRaises(DataValidationError):
            self.indicators.process(invalid_data)
            
        # Test avec données futures sans funding rate
        futures_indicators = MomentumIndicators(
            market_type=MarketType.FUTURES
        )
        with self.assertRaises(DataValidationError):
            futures_indicators.process({'ohlcv': self.test_data})
            
    def test_rsi_calculation(self):
        """Test du calcul du RSI."""
        data = {'ohlcv': self.test_data}
        result = self.indicators.process(data)
        
        # Vérification de la présence du RSI
        self.assertIn('rsi', result)
        rsi_values = result['rsi']
        
        # Test des valeurs du RSI
        self.assertTrue(all(0 <= v <= 100 for v in rsi_values))
        self.assertEqual(len(rsi_values), len(self.test_data))
        
        # Test des signaux RSI
        self.assertIn('rsi_signals', result)
        signals = result['rsi_signals']
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        
    def test_macd_calculation(self):
        """Test du calcul du MACD."""
        data = {'ohlcv': self.test_data}
        result = self.indicators.process(data)
        
        # Vérification des composants du MACD
        self.assertIn('macd', result)
        self.assertIn('macd_signal', result)
        self.assertIn('macd_hist', result)
        
        # Test des longueurs
        self.assertEqual(len(result['macd']), len(self.test_data))
        self.assertEqual(len(result['macd_signal']), len(self.test_data))
        self.assertEqual(len(result['macd_hist']), len(self.test_data))
        
        # Test des signaux MACD
        self.assertIn('macd_signals', result)
        signals = result['macd_signals']
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        
    def test_volume_analysis(self):
        """Test de l'analyse du volume."""
        data = {'ohlcv': self.test_data}
        result = self.indicators.process(data)
        
        # Vérification du profil de volume
        self.assertIn('volume_profile', result)
        volume_profile = result['volume_profile']
        
        # Test des composants du profil
        self.assertIsInstance(volume_profile, VolumeProfile)
        self.assertTrue(hasattr(volume_profile, 'vwap'))
        self.assertTrue(hasattr(volume_profile, 'relative_volume'))
        self.assertTrue(hasattr(volume_profile, 'volume_trend'))
        
        # Test des valeurs
        self.assertTrue(all(v > 0 for v in volume_profile.vwap))
        self.assertTrue(all(v >= 0 for v in volume_profile.relative_volume))
        self.assertTrue(all(v in [-1, 0, 1] for v in volume_profile.volume_trend))
        
    def test_futures_indicators(self):
        """Test des indicateurs spécifiques aux futures."""
        futures_indicators = MomentumIndicators(
            market_type=MarketType.FUTURES
        )
        
        data = {
            'ohlcv': self.test_data,
            'funding_rate': self.funding_data
        }
        
        result = futures_indicators.process(data)
        
        # Vérification des indicateurs futures
        self.assertIn('funding_indicators', result)
        funding = result['funding_indicators']
        
        # Test des composants
        self.assertIn('funding_ma', funding)
        self.assertIn('funding_trend', funding)
        self.assertIn('funding_signal', funding)
        
        # Test des valeurs
        self.assertEqual(len(funding['funding_ma']), len(self.test_data))
        self.assertTrue(all(t in [-1, 0, 1] for t in funding['funding_trend']))
        self.assertTrue(all(s in [-1, 0, 1] for s in funding['funding_signal']))
        
    def test_signal_combination(self):
        """Test de la combinaison des signaux."""
        data = {'ohlcv': self.test_data}
        result = self.indicators.process(data)
        
        # Vérification du signal combiné
        self.assertIn('combined_signal', result)
        signals = result['combined_signal']
        
        # Test des valeurs
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        self.assertEqual(len(signals), len(self.test_data))
        
        # Test de la cohérence avec les sous-signaux
        for i in range(len(signals)):
            if signals[i] != 0:
                # Au moins un des sous-signaux devrait supporter le signal combiné
                self.assertTrue(
                    (result['rsi_signals'][i] == signals[i]) or
                    (result['macd_signals'][i] == signals[i])
                )
                
    def test_error_handling(self):
        """Test de la gestion des erreurs."""
        # Test avec données invalides
        with self.assertRaises(CalculationError):
            self.indicators.process({
                'ohlcv': pd.DataFrame({'close': [np.nan, np.inf, -np.inf]})
            })
            
        # Test avec données vides
        with self.assertRaises(DataValidationError):
            self.indicators.process({
                'ohlcv': pd.DataFrame()
            })
            
        # Test avec fenêtre plus grande que les données
        short_data = pd.DataFrame({
            'close': [1, 2, 3],
            'volume': [100, 200, 300]
        })
        with self.assertRaises(CalculationError):
            self.indicators.process({'ohlcv': short_data})
            
    def test_performance(self):
        """Test des performances de calcul."""
        # Création d'un grand jeu de données
        dates = pd.date_range(
            start='2023-01-01',
            end='2024-01-01',
            freq='1min'
        )
        
        large_data = pd.DataFrame(
            {
                'open': np.random.normal(100, 10, len(dates)),
                'high': np.random.normal(102, 10, len(dates)),
                'low': np.random.normal(98, 10, len(dates)),
                'close': np.random.normal(100, 10, len(dates)),
                'volume': np.random.normal(1000, 100, len(dates))
            },
            index=dates
        )
        
        # Test du temps d'exécution
        import time
        start_time = time.time()
        
        self.indicators.process({'ohlcv': large_data})
        
        execution_time = time.time() - start_time
        
        # Le calcul devrait prendre moins de 1 seconde pour 525600 points (1 an en minutes)
        self.assertLess(execution_time, 1.0)
            
if __name__ == '__main__':
    unittest.main() 