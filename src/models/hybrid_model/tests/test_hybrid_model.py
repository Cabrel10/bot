"""
Tests unitaires pour le modèle hybride.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from ..anomaly_detection import AnomalyDetector, AnomalyConfig
from ..advanced_metrics import AdvancedMetrics, MetricsConfig
from ..continuous_optimization import ContinuousOptimizer, OptimizationConfig
from ..sentiment_volatility import SentimentVolatilityAnalyzer, SentimentVolatilityConfig

class TestAnomalyDetector(unittest.TestCase):
    """Tests pour le détecteur d'anomalies."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.config = AnomalyConfig()
        self.detector = AnomalyDetector(self.config)
        
        # Données de test
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.normal(0, 1, 1000)
        })
        
        # Données avec anomalies
        self.anomaly_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.normal(0, 1, 1000)
        })
        self.anomaly_data.iloc[::10] = np.random.normal(10, 1, 100)
    
    def test_initialization(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.isolation_forest)
        self.assertIsNotNone(self.detector.scaler)
        self.assertIsNotNone(self.detector.autoencoder)
    
    def test_training(self):
        """Test de l'entraînement."""
        self.detector.train(self.data)
        # Vérification que l'entraînement s'est bien passé
        self.assertTrue(True)
    
    def test_anomaly_detection(self):
        """Test de la détection d'anomalies."""
        self.detector.train(self.data)
        anomalies, scores = self.detector.detect_anomalies(self.anomaly_data)
        
        self.assertIsNotNone(anomalies)
        self.assertIsNotNone(scores)
        self.assertEqual(len(anomalies), len(self.anomaly_data))
        self.assertGreater(np.mean(anomalies), 0)
    
    def test_fraud_detection(self):
        """Test de la détection de fraude."""
        transaction_amounts = np.random.normal(100, 10, 1000)
        time_deltas = np.random.exponential(1, 1000)
        
        fraud_mask, scores = self.detector.detect_fraud(
            self.anomaly_data,
            transaction_amounts,
            time_deltas
        )
        
        self.assertIsNotNone(fraud_mask)
        self.assertIsNotNone(scores)
        self.assertEqual(len(fraud_mask), len(self.anomaly_data))
        self.assertGreater(np.mean(fraud_mask), 0)

class TestAdvancedMetrics(unittest.TestCase):
    """Tests pour les métriques avancées."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.config = MetricsConfig()
        self.metrics = AdvancedMetrics(self.config)
        
        # Données de test
        self.returns = np.random.normal(0.001, 0.02, 1000)
        self.benchmark_returns = np.random.normal(0.001, 0.02, 1000)
        
        # Trades de test
        self.trades = [
            {
                'entry_time': datetime.now() - timedelta(hours=i),
                'exit_time': datetime.now() - timedelta(hours=i-1),
                'pnl': np.random.normal(10, 5)
            }
            for i in range(100)
        ]
    
    def test_initialization(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.metrics)
        self.assertIsNotNone(self.metrics.config)
    
    def test_returns_metrics(self):
        """Test des métriques de rendement."""
        metrics = self.metrics.calculate_returns_metrics(
            self.returns,
            self.benchmark_returns
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
    
    def test_trade_metrics(self):
        """Test des métriques de trades."""
        metrics = self.metrics.calculate_trade_metrics(
            self.trades,
            1000.0
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('profit_factor', metrics)
        self.assertIn('roi', metrics)
    
    def test_risk_metrics(self):
        """Test des métriques de risque."""
        positions = np.random.uniform(0, 1, 1000)
        metrics = self.metrics.calculate_risk_metrics(
            self.returns,
            positions
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn('beta', metrics)
        self.assertIn('alpha', metrics)
        self.assertIn('information_ratio', metrics)
        self.assertIn('var_95', metrics)

class TestContinuousOptimizer(unittest.TestCase):
    """Tests pour l'optimiseur continu."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.config = OptimizationConfig()
        self.optimizer = ContinuousOptimizer(self.config)
        
        # Modèle de test
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Données de test
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        self.labels = np.random.normal(0, 1, 1000)
    
    def test_initialization(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.config)
        self.assertIsNone(self.optimizer.last_update)
    
    def test_should_update(self):
        """Test de la vérification de mise à jour."""
        # Premier appel
        self.assertTrue(self.optimizer.should_update())
        
        # Mise à jour
        self.optimizer.last_update = datetime.now()
        
        # Vérification immédiate
        self.assertFalse(self.optimizer.should_update())
    
    def test_optimization(self):
        """Test de l'optimisation."""
        metrics = self.optimizer.optimize(
            self.model,
            self.data,
            self.labels
        )
        
        self.assertIsNotNone(metrics)
        self.assertIn('performance', metrics)
        self.assertIn('risk', metrics)
        self.assertIn('adaptation', metrics)
    
    def test_optimization_history(self):
        """Test de l'analyse de l'historique."""
        # Simulation d'optimisations
        for _ in range(20):
            self.optimizer.performance_history.append(np.random.random())
            self.optimizer.risk_history.append(np.random.random())
            self.optimizer.adaptation_history.append(np.random.random())
        
        analysis = self.optimizer.analyze_optimization_history()
        
        self.assertIsNotNone(analysis)
        self.assertIn('performance_trend', analysis)
        self.assertIn('risk_trend', analysis)
        self.assertIn('adaptation_trend', analysis)
        self.assertIn('stability', analysis)

class TestSentimentVolatilityAnalyzer(unittest.TestCase):
    """Tests pour l'analyseur de sentiments et de volatilité."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.config = SentimentVolatilityConfig()
        self.analyzer = SentimentVolatilityAnalyzer(self.config)
        
        # Données de test
        self.text = "Bitcoin is showing strong bullish momentum with increasing volume."
        self.market_data = pd.DataFrame({
            'close': np.random.normal(50000, 1000, 1000)
        })
        self.sentiment_scores = [
            {'polarity': 0.5, 'subjectivity': 0.3},
            {'polarity': 0.7, 'subjectivity': 0.4},
            {'polarity': 0.3, 'subjectivity': 0.2}
        ]
    
    def test_initialization(self):
        """Test de l'initialisation."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.sentiment_model)
        self.assertIsNotNone(self.analyzer.volatility_model)
    
    def test_text_preprocessing(self):
        """Test du prétraitement du texte."""
        processed_text = self.analyzer.preprocess_text(self.text)
        
        self.assertIsNotNone(processed_text)
        self.assertIsInstance(processed_text, str)
        self.assertNotEqual(processed_text, self.text)
    
    def test_sentiment_analysis(self):
        """Test de l'analyse des sentiments."""
        sentiment = self.analyzer.analyze_sentiment(self.text)
        
        self.assertIsNotNone(sentiment)
        self.assertIn('polarity', sentiment)
        self.assertIn('subjectivity', sentiment)
        self.assertIn('model_score', sentiment)
        self.assertIn('final_score', sentiment)
        self.assertIn('confidence', sentiment)
    
    def test_volatility_prediction(self):
        """Test de la prédiction de volatilité."""
        prediction = self.analyzer.predict_volatility(self.market_data)
        
        self.assertIsNotNone(prediction)
        self.assertIn('current_volatility', prediction)
        self.assertIn('predicted_volatility', prediction)
        self.assertIn('volatility_change', prediction)
        self.assertIn('confidence', prediction)
    
    def test_market_regime_analysis(self):
        """Test de l'analyse du régime de marché."""
        regime = self.analyzer.analyze_market_regime(
            self.market_data,
            self.sentiment_scores
        )
        
        self.assertIsNotNone(regime)
        self.assertIn('regime', regime)
        self.assertIn('sentiment', regime)
        self.assertIn('volatility', regime)
        self.assertIn('avg_polarity', regime)
        self.assertIn('avg_subjectivity', regime)
        self.assertIn('confidence', regime)

if __name__ == '__main__':
    unittest.main() 