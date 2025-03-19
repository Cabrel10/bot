"""Tests unitaires pour le module training_parameters."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.core.models.training_parameters import TemporalParameters

class MockModel:
    """Modèle fictif pour les tests."""
    
    def train(self, X, y):
        """Simule l'entraînement."""
        pass
        
    def evaluate(self, X, y) -> float:
        """Simule l'évaluation."""
        return 0.8

class TestTemporalParameters(unittest.TestCase):
    """Tests pour la classe TemporalParameters."""
    
    def setUp(self):
        """Initialise les données pour les tests."""
        self.params = TemporalParameters(
            window_size=10,
            update_frequency='1D',
            prediction_horizon=5,
            train_test_split=0.8
        )
        
        # Création d'un DataFrame de test
        dates = pd.date_range(
            start='2023-01-01',
            end='2023-01-31',
            freq='D'
        )
        self.test_data = pd.DataFrame(
            np.random.randn(len(dates), 3),
            index=dates,
            columns=['price', 'volume', 'volatility']
        )
        
    def test_initialization(self):
        """Teste l'initialisation des paramètres."""
        self.assertEqual(self.params.window_size, 10)
        self.assertEqual(self.params.update_frequency, '1D')
        self.assertEqual(self.params.prediction_horizon, 5)
        self.assertEqual(self.params.train_test_split, 0.8)
        
    def test_invalid_initialization(self):
        """Teste la validation des paramètres d'initialisation."""
        with self.assertRaises(ValueError):
            TemporalParameters(window_size=0)
            
        with self.assertRaises(ValueError):
            TemporalParameters(train_test_split=1.5)
            
        with self.assertRaises(ValueError):
            TemporalParameters(prediction_horizon=0)
            
    def test_create_rolling_windows(self):
        """Teste la création des fenêtres glissantes."""
        X, y = self.params.create_rolling_windows(self.test_data)
        
        # Vérification des dimensions
        expected_samples = len(self.test_data) - self.params.window_size - self.params.prediction_horizon + 1
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], self.params.window_size)
        self.assertEqual(X.shape[2], self.test_data.shape[1])
        self.assertEqual(y.shape[0], expected_samples)
        self.assertEqual(y.shape[1], self.params.prediction_horizon)
        
        # Vérification des valeurs
        first_window = X[0]
        self.assertTrue(np.array_equal(
            first_window,
            self.test_data.iloc[:self.params.window_size].values
        ))
        
    def test_split_data(self):
        """Teste la division des données."""
        X = np.random.randn(100, 10, 3)
        y = np.random.randn(100, 5)
        
        X_train, X_test, y_train, y_test = self.params.split_data(X, y)
        
        # Vérification des dimensions
        self.assertEqual(len(X_train), 80)  # 80% des données
        self.assertEqual(len(X_test), 20)   # 20% des données
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Vérification de la continuité
        self.assertTrue(np.array_equal(X[:80], X_train))
        self.assertTrue(np.array_equal(X[80:], X_test))
        
    def test_create_time_windows(self):
        """Teste la création des fenêtres temporelles."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        windows = self.params.create_time_windows(start_date, end_date)
        
        # Vérification du nombre de fenêtres
        expected_windows = len(pd.date_range(start_date, end_date, freq=self.params.update_frequency))
        self.assertEqual(len(windows), expected_windows)
        
        # Vérification de la première fenêtre
        first_window = windows[0]
        self.assertEqual(first_window[0], start_date)
        self.assertEqual(
            first_window[1],
            min(start_date + timedelta(days=self.params.window_size), end_date)
        )
        
    def test_evaluate_window_sizes(self):
        """Teste l'évaluation des tailles de fenêtres."""
        model = MockModel()
        window_sizes = [5, 10, 15]
        
        results = self.params.evaluate_window_sizes(
            self.test_data,
            model,
            window_sizes
        )
        
        # Vérification des résultats
        self.assertEqual(len(results), len(window_sizes))
        for size in window_sizes:
            self.assertIn(size, results)
            self.assertEqual(results[size], 0.8)  # Valeur de retour du mock
            
        # Vérification que la taille originale est restaurée
        self.assertEqual(self.params.window_size, 10)
        
    def test_optimize_parameters(self):
        """Teste l'optimisation des paramètres."""
        model = MockModel()
        param_ranges = {
            'window_size': [5, 10],
            'update_frequency': ['1D', '2D'],
            'prediction_horizon': [3, 5]
        }
        
        best_params = self.params.optimize_parameters(
            self.test_data,
            model,
            param_ranges
        )
        
        # Vérification des résultats
        self.assertIn('window_size', best_params)
        self.assertIn('update_frequency', best_params)
        self.assertIn('prediction_horizon', best_params)
        
        # Vérification que les paramètres originaux sont restaurés
        self.assertEqual(self.params.window_size, 10)
        self.assertEqual(self.params.update_frequency, '1D')
        self.assertEqual(self.params.prediction_horizon, 5)
        
    def test_empty_data(self):
        """Teste le comportement avec des données vides."""
        empty_data = pd.DataFrame(columns=['price'])
        
        with self.assertRaises(ValueError):
            self.params.create_rolling_windows(empty_data)
            
    def test_invalid_data(self):
        """Teste le comportement avec des données invalides."""
        invalid_data = pd.DataFrame(
            np.random.randn(5, 3),  # Trop peu de données
            columns=['price', 'volume', 'volatility']
        )
        
        with self.assertRaises(ValueError):
            self.params.create_rolling_windows(invalid_data)
            
    def test_missing_values(self):
        """Teste le comportement avec des valeurs manquantes."""
        data_with_nan = self.test_data.copy()
        data_with_nan.iloc[5:10, 0] = np.nan
        
        with self.assertRaises(ValueError):
            self.params.create_rolling_windows(data_with_nan)

if __name__ == '__main__':
    unittest.main() 