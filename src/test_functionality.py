"""
Script de test des fonctionnalités du modèle hybride.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import tensorflow as tf
from models.hybrid_model.hybrid_model import HybridModel, HybridConfig, CNNConfig, GNAConfig
from models.hybrid_model.anomaly_detection import AnomalyDetector, AnomalyConfig
from models.hybrid_model.continuous_optimization import ContinuousOptimizer, OptimizationConfig
from models.hybrid_model.sentiment_volatility import SentimentVolatilityAnalyzer, SentimentVolatilityConfig

# Configuration GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erreur lors de la configuration GPU : {e}")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Génère des données de test."""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='H')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, n_samples),
        'high': np.random.normal(102, 10, n_samples),
        'low': np.random.normal(98, 10, n_samples),
        'close': np.random.normal(101, 10, n_samples),
        'volume': np.random.exponential(1000, n_samples)
    }, index=dates)
    
    # Ajout de tendances
    data['close'] = data['close'].cumsum() / 20 + 100
    return data

def test_hybrid_model():
    """Teste le modèle hybride."""
    logger.info("Test du modèle hybride...")
    
    # Configuration
    cnn_config = CNNConfig(
        input_shape=(60, 5),
        num_classes=3
    )
    
    gna_config = GNAConfig(
        population_size=50,
        num_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=5,
        risk_constraints={'max_drawdown': 0.2},
        fitness_weights={'returns': 0.7, 'risk': 0.3}
    )
    
    hybrid_config = HybridConfig(
        cnn_config=cnn_config,
        gna_config=gna_config
    )
    
    # Création du modèle
    model = HybridModel(hybrid_config)
    
    # Données de test
    data = generate_test_data()
    labels = np.random.randint(0, 3, len(data))
    
    # Entraînement
    model.train(data, labels)
    
    # Prédiction
    predictions = model.predict(data.iloc[-100:])
    logger.info(f"Prédictions générées: {predictions.shape}")

def test_anomaly_detection():
    """Teste la détection d'anomalies."""
    logger.info("Test de la détection d'anomalies...")
    
    config = AnomalyConfig()
    detector = AnomalyDetector(config)
    
    # Données de test
    data = generate_test_data()
    
    # Entraînement
    detector.train(data)
    
    # Détection
    anomalies, scores = detector.detect_anomalies(data)
    logger.info(f"Anomalies détectées: {np.sum(anomalies)}")

def test_continuous_optimization():
    """Teste l'optimisation continue."""
    logger.info("Test de l'optimisation continue...")
    
    config = OptimizationConfig()
    optimizer = ContinuousOptimizer(config)
    
    # Création d'un modèle simple pour le test
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    # Données de test
    data = generate_test_data()
    labels = np.random.normal(0, 1, len(data))
    
    # Optimisation
    metrics = optimizer.optimize(model, data, labels)
    logger.info(f"Métriques d'optimisation: {metrics}")

def test_sentiment_volatility():
    """Teste l'analyse des sentiments et de la volatilité."""
    logger.info("Test de l'analyse des sentiments et de la volatilité...")
    
    config = SentimentVolatilityConfig()
    analyzer = SentimentVolatilityAnalyzer(config)
    
    # Test de l'analyse des sentiments
    text = "Bitcoin shows strong bullish momentum with increasing volume"
    sentiment = analyzer.analyze_sentiment(text)
    logger.info(f"Analyse des sentiments: {sentiment}")
    
    # Test de la prédiction de volatilité
    data = generate_test_data()
    volatility = analyzer.predict_volatility(data)
    logger.info(f"Prédiction de volatilité: {volatility}")

def main():
    """Fonction principale de test."""
    logger.info("Début des tests de fonctionnalités...")
    
    try:
        test_hybrid_model()
        test_anomaly_detection()
        test_continuous_optimization()
        test_sentiment_volatility()
        logger.info("Tous les tests ont réussi !")
        
    except Exception as e:
        logger.error(f"Erreur lors des tests: {e}")
        raise

if __name__ == "__main__":
    main() 