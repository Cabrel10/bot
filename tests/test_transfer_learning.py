"""Tests unitaires pour le module transfer_learning."""

import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

from src.core.models.transfer_learning import TransferConfig, ModelAdapter
from src.core.models.neural_network import TradingNeuralNetwork

class MockTradingNeuralNetwork:
    """Mock pour TradingNeuralNetwork."""
    
    def __init__(self):
        self.config = {'mock': True}
        
    def get_config(self):
        return self.config
        
    def predict(self, X):
        return np.random.randn(len(X))
        
    def clone(self):
        return MockTradingNeuralNetwork()
        
    def get_layer(self, name):
        mock_layer = Mock()
        mock_layer.parameters = Mock(return_value=[Mock()])
        return mock_layer
        
    def set_learning_rate(self, lr):
        pass
        
    def train_epoch(self, X, y, batch_size):
        return {'loss': 0.1}
        
    def validate(self, X, y):
        return {'loss': 0.2}
        
    def save_checkpoint(self, path):
        pass
        
    def load_checkpoint(self, path):
        pass

class TestTransferConfig(unittest.TestCase):
    """Tests pour la classe TransferConfig."""
    
    def setUp(self):
        """Initialise les configurations pour les tests."""
        self.config = TransferConfig(
            source_model_path='models/source.pt',
            target_data_ratio=0.3,
            fine_tuning_epochs=10,
            learning_rate=0.001,
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=3,
            layers_to_freeze=['layer1', 'layer2'],
            adaptation_threshold=0.15
        )
        
    def test_valid_config(self):
        """Teste une configuration valide."""
        self.assertEqual(self.config.source_model_path, 'models/source.pt')
        self.assertEqual(self.config.target_data_ratio, 0.3)
        self.assertEqual(self.config.fine_tuning_epochs, 10)
        self.assertEqual(self.config.learning_rate, 0.001)
        
    def test_invalid_config(self):
        """Teste la validation des paramètres invalides."""
        with self.assertRaises(ValueError):
            TransferConfig(
                source_model_path='',
                target_data_ratio=1.5
            )
            
        with self.assertRaises(ValueError):
            TransferConfig(
                source_model_path='model.pt',
                fine_tuning_epochs=0
            )

class TestModelAdapter(unittest.TestCase):
    """Tests pour la classe ModelAdapter."""
    
    def setUp(self):
        """Initialise les données pour les tests."""
        self.config = TransferConfig(
            source_model_path='models/source.pt',
            target_data_ratio=0.3,
            fine_tuning_epochs=3,  # Réduit pour les tests
            learning_rate=0.001,
            batch_size=32
        )
        
        self.adapter = ModelAdapter(self.config)
        self.source_model = MockTradingNeuralNetwork()
        
        # Données de test
        self.target_data = {
            'X_train': np.random.randn(100, 10, 5),
            'y_train': np.random.randn(100),
            'X_val': np.random.randn(20, 10, 5),
            'y_val': np.random.randn(20)
        }
        
    async def test_adapt_model_no_adaptation_needed(self):
        """Teste le cas où l'adaptation n'est pas nécessaire."""
        # Configuration pour ne pas nécessiter d'adaptation
        self.adapter.config.adaptation_threshold = 1.0
        
        adapted_model = await self.adapter.adapt_model(
            self.source_model,
            self.target_data
        )
        
        self.assertEqual(
            adapted_model.get_config(),
            self.source_model.get_config()
        )
        
    async def test_adapt_model_with_adaptation(self):
        """Teste l'adaptation complète du modèle."""
        # Configuration pour forcer l'adaptation
        self.adapter.config.adaptation_threshold = 0.0
        
        adapted_model = await self.adapter.adapt_model(
            self.source_model,
            self.target_data
        )
        
        # Vérification de l'historique
        history = self.adapter.get_adaptation_history()
        self.assertEqual(len(history), 1)
        self.assertIn('initial_performance', history[0])
        self.assertIn('final_performance', history[0])
        
    async def test_evaluate_model(self):
        """Teste l'évaluation du modèle."""
        metrics = await self.adapter._evaluate_model(
            self.source_model,
            self.target_data
        )
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('directional_accuracy', metrics)
        
    def test_needs_adaptation(self):
        """Teste la détection du besoin d'adaptation."""
        # Performance au-dessus du seuil
        high_performance = {'metric': 0.2}
        self.assertTrue(
            self.adapter._needs_adaptation(high_performance)
        )
        
        # Performance en dessous du seuil
        low_performance = {'metric': 0.1}
        self.assertFalse(
            self.adapter._needs_adaptation(low_performance)
        )
        
    async def test_prepare_model_for_adaptation(self):
        """Teste la préparation du modèle pour l'adaptation."""
        self.adapter.config.layers_to_freeze = ['layer1']
        
        adapted_model = await self.adapter._prepare_model_for_adaptation(
            self.source_model
        )
        
        self.assertIsNotNone(adapted_model)
        self.assertNotEqual(id(adapted_model), id(self.source_model))
        
    async def test_fine_tune(self):
        """Teste le fine-tuning du modèle."""
        model = await self.adapter._fine_tune(
            self.source_model,
            self.target_data
        )
        
        # Vérification de l'historique d'adaptation
        self.assertIsNotNone(
            self.adapter._current_adaptation.get('training_history')
        )
        
    def test_calculate_directional_accuracy(self):
        """Teste le calcul de la précision directionnelle."""
        predictions = np.array([1, 2, 1, 3])
        targets = np.array([1, 3, 1, 2])
        
        accuracy = self.adapter._calculate_directional_accuracy(
            predictions,
            targets
        )
        
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)
        
    def test_record_adaptation(self):
        """Teste l'enregistrement de l'adaptation."""
        self.adapter._current_adaptation = {
            'start_time': datetime.now(),
            'initial_performance': {'metric': 0.5}
        }
        
        self.adapter._record_adaptation()
        
        history = self.adapter.get_adaptation_history()
        self.assertEqual(len(history), 1)
        self.assertIn('duration', history[0])
        
    def test_get_performance_evolution(self):
        """Teste la récupération de l'évolution des performances."""
        # Simulation de deux adaptations
        adaptation1 = {
            'final_performance': {'metric1': 0.8, 'metric2': 0.6}
        }
        adaptation2 = {
            'final_performance': {'metric1': 0.9, 'metric2': 0.7}
        }
        
        self.adapter._adaptation_history = [adaptation1, adaptation2]
        
        evolution = self.adapter.get_performance_evolution()
        
        self.assertIn('metric1', evolution)
        self.assertIn('metric2', evolution)
        self.assertEqual(len(evolution['metric1']), 2)
        self.assertEqual(evolution['metric1'], [0.8, 0.9])
        
    def test_error_handling(self):
        """Teste la gestion des erreurs."""
        async def test_invalid_data():
            with self.assertRaises(ValueError):
                await self.adapter.adapt_model(
                    self.source_model,
                    {'invalid': 'data'}
                )
                
        async def test_invalid_model():
            with self.assertRaises(ValueError):
                await self.adapter.adapt_model(
                    None,
                    self.target_data
                )
                
        # Exécution des tests asynchrones
        import asyncio
        asyncio.run(test_invalid_data())
        asyncio.run(test_invalid_model())

if __name__ == '__main__':
    unittest.main() 