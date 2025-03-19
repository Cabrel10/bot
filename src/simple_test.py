"""
Script de test simplifié pour vérifier les imports.
"""

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime, timedelta

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
    
    # Ajouter des tendances
    data['close'] = data['close'].cumsum() / 20 + 100
    return data

def main():
    """Fonction principale."""
    logger.info("Démarrage du test simplifié...")
    
    # Génération des données de test
    data = generate_test_data()
    logger.info(f"Données générées: {data.shape}")
    
    # Test simple de TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # Compilation du modèle
    model.compile(optimizer='adam', loss='mse')
    
    # Préparation des données pour TensorFlow
    X = data[['open', 'high', 'low', 'close', 'volume']].values
    y = data['close'].shift(-1).fillna(method='ffill').values
    
    # Entraînement rapide
    logger.info("Entraînement d'un modèle simple...")
    model.fit(X, y, epochs=2, batch_size=32, verbose=1, validation_split=0.2)
    
    logger.info("Test simplifié terminé avec succès!")

if __name__ == "__main__":
    main() 