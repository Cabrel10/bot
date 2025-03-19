"""
Script de test pour le module de détection d'anomalies.
"""

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime, timedelta
from models.hybrid_model.anomaly_detection import AnomalyDetector, AnomalyConfig

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Génère des données de test."""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='h')
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
        'feature5': np.random.normal(0, 1, n_samples)
    }, index=dates)
    
    # Ajouter des anomalies
    for i in range(50):
        idx = np.random.randint(0, n_samples)
        data.iloc[idx] = data.iloc[idx] * 10
    
    return data

def main():
    """Fonction principale."""
    logger.info("Test du module de détection d'anomalies...")
    
    # Configuration
    config = AnomalyConfig(
        isolation_forest_contamination=0.05,
        z_score_threshold=3.0,
        autoencoder_latent_dim=16,
        reconstruction_threshold=0.1,
        min_samples_for_training=100
    )
    
    # Création du détecteur
    detector = AnomalyDetector(config)
    logger.info("Détecteur d'anomalies créé.")
    
    # Données de test
    data = generate_test_data(1000)
    logger.info(f"Données générées: {data.shape}")
    
    # Entraînement
    logger.info("Entraînement du détecteur...")
    detector.train(data)
    
    # Détection
    logger.info("Détection des anomalies...")
    anomalies, scores = detector.detect_anomalies(data)
    
    # Résultats
    logger.info(f"Nombre d'anomalies détectées: {np.sum(anomalies)}")
    logger.info(f"Scores: {scores}")
    
    logger.info("Test du module de détection d'anomalies terminé.")

if __name__ == "__main__":
    main() 