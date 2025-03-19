"""
Détection d'anomalies et de fraude pour le modèle hybride.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers, models

@dataclass
class AnomalyConfig:
    """Configuration pour la détection d'anomalies."""
    isolation_forest_contamination: float = 0.1
    z_score_threshold: float = 3.0
    autoencoder_latent_dim: int = 32
    reconstruction_threshold: float = 0.1
    min_samples_for_training: int = 1000

class AnomalyDetector:
    """Détecteur d'anomalies et de fraude."""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.isolation_forest = IsolationForest(
            contamination=config.isolation_forest_contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.autoencoder = self._build_autoencoder()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _build_autoencoder(self) -> models.Model:
        """Construit l'autoencodeur pour la détection d'anomalies."""
        input_dim = 10  # À ajuster selon les données
        
        # Encoder
        encoder = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.config.autoencoder_latent_dim, activation='relu')
        ])
        
        # Decoder
        decoder = models.Sequential([
            layers.Dense(32, activation='relu', 
                        input_shape=(self.config.autoencoder_latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        
        # Autoencoder
        autoencoder = models.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train(self, data: pd.DataFrame):
        """
        Entraîne les détecteurs d'anomalies.
        
        Args:
            data: Données d'entraînement
        """
        try:
            if len(data) < self.config.min_samples_for_training:
                self.logger.warning(
                    f"Pas assez d'échantillons pour l'entraînement: {len(data)} < {self.config.min_samples_for_training}"
                )
                return
            
            # Normalisation des données
            scaled_data = self.scaler.fit_transform(data)
            
            # Entraînement de l'Isolation Forest
            self.isolation_forest.fit(scaled_data)
            
            # Entraînement de l'autoencodeur
            self.autoencoder.fit(
                scaled_data, scaled_data,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.logger.info("Détecteurs d'anomalies entraînés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement des détecteurs: {e}")
            raise
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        method: str = "ensemble"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Détecte les anomalies dans les données.
        
        Args:
            data: Données à analyser
            method: Méthode de détection (ensemble, isolation_forest, autoencoder, z_score)
            
        Returns:
            Masque des anomalies et scores
        """
        try:
            # Normalisation des données
            scaled_data = self.scaler.transform(data)
            
            if method == "ensemble":
                return self._ensemble_detection(scaled_data)
            elif method == "isolation_forest":
                return self._isolation_forest_detection(scaled_data)
            elif method == "autoencoder":
                return self._autoencoder_detection(scaled_data)
            else:  # z_score
                return self._z_score_detection(scaled_data)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection d'anomalies: {e}")
            raise
    
    def _ensemble_detection(
        self,
        scaled_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détection d'anomalies par ensemble de méthodes."""
        # Isolation Forest
        if_anomalies = self._isolation_forest_detection(scaled_data)[0]
        
        # Autoencoder
        ae_anomalies = self._autoencoder_detection(scaled_data)[0]
        
        # Z-score
        z_anomalies = self._z_score_detection(scaled_data)[0]
        
        # Combinaison des résultats
        ensemble_anomalies = np.any([if_anomalies, ae_anomalies, z_anomalies], axis=0)
        
        # Calcul des scores
        scores = {
            'isolation_forest': np.mean(if_anomalies),
            'autoencoder': np.mean(ae_anomalies),
            'z_score': np.mean(z_anomalies),
            'ensemble': np.mean(ensemble_anomalies)
        }
        
        return ensemble_anomalies, scores
    
    def _isolation_forest_detection(
        self,
        scaled_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détection d'anomalies par Isolation Forest."""
        predictions = self.isolation_forest.predict(scaled_data)
        anomalies = predictions == -1
        
        scores = {
            'isolation_forest': np.mean(anomalies),
            'scores': self.isolation_forest.score_samples(scaled_data)
        }
        
        return anomalies, scores
    
    def _autoencoder_detection(
        self,
        scaled_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détection d'anomalies par autoencodeur."""
        # Reconstruction
        reconstructed = self.autoencoder.predict(scaled_data)
        
        # Calcul de l'erreur de reconstruction
        reconstruction_error = np.mean(np.square(scaled_data - reconstructed), axis=1)
        
        # Détection des anomalies
        anomalies = reconstruction_error > self.config.reconstruction_threshold
        
        scores = {
            'autoencoder': np.mean(anomalies),
            'reconstruction_error': reconstruction_error
        }
        
        return anomalies, scores
    
    def _z_score_detection(
        self,
        scaled_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détection d'anomalies par Z-score."""
        z_scores = np.abs(stats.zscore(scaled_data, axis=0))
        anomalies = np.any(z_scores > self.config.z_score_threshold, axis=1)
        
        scores = {
            'z_score': np.mean(anomalies),
            'z_scores': z_scores
        }
        
        return anomalies, scores
    
    def detect_fraud(
        self,
        data: pd.DataFrame,
        transaction_amounts: np.ndarray,
        time_deltas: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Détecte les transactions frauduleuses.
        
        Args:
            data: Données des transactions
            transaction_amounts: Montants des transactions
            time_deltas: Délais entre les transactions
            
        Returns:
            Masque des transactions frauduleuses et scores
        """
        try:
            # Détection des anomalies dans les montants
            amount_anomalies, amount_scores = self._detect_amount_anomalies(transaction_amounts)
            
            # Détection des anomalies dans les délais
            time_anomalies, time_scores = self._detect_time_anomalies(time_deltas)
            
            # Détection des anomalies dans les données
            data_anomalies, data_scores = self.detect_anomalies(data)
            
            # Combinaison des résultats
            fraud_mask = np.any([amount_anomalies, time_anomalies, data_anomalies], axis=0)
            
            # Calcul des scores
            scores = {
                'amount_anomalies': np.mean(amount_anomalies),
                'time_anomalies': np.mean(time_anomalies),
                'data_anomalies': np.mean(data_anomalies),
                'fraud_rate': np.mean(fraud_mask)
            }
            
            return fraud_mask, scores
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de fraude: {e}")
            raise
    
    def _detect_amount_anomalies(
        self,
        amounts: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détecte les anomalies dans les montants des transactions."""
        # Calcul des statistiques
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # Détection des anomalies
        z_scores = np.abs((amounts - mean_amount) / std_amount)
        anomalies = z_scores > self.config.z_score_threshold
        
        scores = {
            'amount_anomalies': np.mean(anomalies),
            'z_scores': z_scores
        }
        
        return anomalies, scores
    
    def _detect_time_anomalies(
        self,
        time_deltas: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Détecte les anomalies dans les délais entre transactions."""
        # Calcul des statistiques
        mean_delta = np.mean(time_deltas)
        std_delta = np.std(time_deltas)
        
        # Détection des anomalies
        z_scores = np.abs((time_deltas - mean_delta) / std_delta)
        anomalies = z_scores > self.config.z_score_threshold
        
        scores = {
            'time_anomalies': np.mean(anomalies),
            'z_scores': z_scores
        }
        
        return anomalies, scores
    
    def analyze_anomaly_patterns(
        self,
        data: pd.DataFrame,
        anomalies: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyse les patterns d'anomalies.
        
        Args:
            data: Données
            anomalies: Masque des anomalies
            
        Returns:
            Statistiques sur les patterns d'anomalies
        """
        try:
            # Analyse par caractéristique
            feature_stats = {}
            for col in data.columns:
                feature_stats[col] = {
                    'mean': np.mean(data[col][anomalies]),
                    'std': np.std(data[col][anomalies]),
                    'min': np.min(data[col][anomalies]),
                    'max': np.max(data[col][anomalies])
                }
            
            # Analyse temporelle
            temporal_stats = {
                'hour_distribution': np.histogram(
                    pd.to_datetime(data.index).hour[anomalies],
                    bins=24,
                    range=(0, 24)
                )[0],
                'day_distribution': np.histogram(
                    pd.to_datetime(data.index).dayofweek[anomalies],
                    bins=7,
                    range=(0, 7)
                )[0]
            }
            
            return {
                'feature_stats': feature_stats,
                'temporal_stats': temporal_stats,
                'anomaly_rate': np.mean(anomalies)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des patterns: {e}")
            raise 