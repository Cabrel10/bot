"""
Optimisation continue du modèle hybride.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from datetime import datetime, timedelta

@dataclass
class OptimizationConfig:
    """Configuration pour l'optimisation continue."""
    update_frequency: int = 24  # heures
    min_samples: int = 1000
    validation_split: float = 0.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    momentum: float = 0.9
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    max_optimization_time: int = 3600  # secondes
    performance_threshold: float = 0.1
    risk_threshold: float = 0.2
    adaptation_threshold: float = 0.15

class ContinuousOptimizer:
    """Optimiseur continu du modèle hybride."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.scaler = StandardScaler()
        self._setup_logging()
        self.last_update = None
        self.performance_history = []
        self.risk_history = []
        self.adaptation_history = []
        
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
    
    def should_update(self) -> bool:
        """
        Vérifie si le modèle doit être mis à jour.
        
        Returns:
            True si une mise à jour est nécessaire
        """
        if self.last_update is None:
            return True
        
        time_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        return time_since_update >= self.config.update_frequency
    
    def optimize(
        self,
        model: tf.keras.Model,
        data: pd.DataFrame,
        labels: np.ndarray,
        validation_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Optimise le modèle avec les nouvelles données.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            labels: Labels d'entraînement
            validation_data: Données de validation (optionnel)
            
        Returns:
            Métriques d'optimisation
        """
        try:
            if len(data) < self.config.min_samples:
                self.logger.warning(
                    f"Pas assez d'échantillons pour l'optimisation: {len(data)} < {self.config.min_samples}"
                )
                return {}
            
            # Préparation des données
            scaled_data = self.scaler.fit_transform(data)
            
            # Configuration de l'optimiseur
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum
            )
            
            # Compilation du modèle
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.min_delta,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            # Entraînement
            start_time = datetime.now()
            history = model.fit(
                scaled_data,
                labels,
                validation_split=self.config.validation_split,
                validation_data=validation_data,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            # Vérification du temps d'optimisation
            optimization_time = (datetime.now() - start_time).total_seconds()
            if optimization_time > self.config.max_optimization_time:
                self.logger.warning(
                    f"Temps d'optimisation dépassé: {optimization_time} > {self.config.max_optimization_time}"
                )
            
            # Calcul des métriques
            metrics = self._calculate_optimization_metrics(history)
            
            # Mise à jour de l'historique
            self.last_update = datetime.now()
            self.performance_history.append(metrics['performance'])
            self.risk_history.append(metrics['risk'])
            self.adaptation_history.append(metrics['adaptation'])
            
            # Vérification des seuils
            if self._check_thresholds(metrics):
                self.logger.info("Optimisation réussie")
            else:
                self.logger.warning("Optimisation sous-optimale")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation: {e}")
            raise
    
    def _calculate_optimization_metrics(
        self,
        history: tf.keras.callbacks.History
    ) -> Dict[str, float]:
        """
        Calcule les métriques d'optimisation.
        
        Args:
            history: Historique d'entraînement
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {}
        
        # Performance
        metrics['performance'] = 1 - history.history['val_loss'][-1]
        
        # Risque
        metrics['risk'] = np.std(history.history['val_loss'])
        
        # Adaptation
        if len(self.performance_history) > 0:
            metrics['adaptation'] = (
                metrics['performance'] - self.performance_history[-1]
            ) / self.performance_history[-1]
        else:
            metrics['adaptation'] = 0
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Vérifie si les métriques respectent les seuils.
        
        Args:
            metrics: Métriques à vérifier
            
        Returns:
            True si tous les seuils sont respectés
        """
        return (
            metrics['performance'] >= self.config.performance_threshold and
            metrics['risk'] <= self.config.risk_threshold and
            abs(metrics['adaptation']) <= self.config.adaptation_threshold
        )
    
    def analyze_optimization_history(
        self,
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Analyse l'historique d'optimisation.
        
        Args:
            window_size: Taille de la fenêtre glissante
            
        Returns:
            Statistiques sur l'historique
        """
        try:
            if len(self.performance_history) < window_size:
                self.logger.warning(
                    f"Pas assez de données pour l'analyse: {len(self.performance_history)} < {window_size}"
                )
                return {}
            
            # Calcul des moyennes mobiles
            performance_ma = pd.Series(self.performance_history).rolling(window=window_size).mean()
            risk_ma = pd.Series(self.risk_history).rolling(window=window_size).mean()
            adaptation_ma = pd.Series(self.adaptation_history).rolling(window=window_size).mean()
            
            # Calcul des tendances
            performance_trend = self._calculate_trend(performance_ma)
            risk_trend = self._calculate_trend(risk_ma)
            adaptation_trend = self._calculate_trend(adaptation_ma)
            
            # Calcul de la stabilité
            stability = self._calculate_stability()
            
            return {
                'performance_trend': performance_trend,
                'risk_trend': risk_trend,
                'adaptation_trend': adaptation_trend,
                'stability': stability,
                'last_performance': self.performance_history[-1],
                'last_risk': self.risk_history[-1],
                'last_adaptation': self.adaptation_history[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de l'historique: {e}")
            raise
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        Calcule la tendance d'une série.
        
        Args:
            series: Série à analyser
            
        Returns:
            Coefficient de tendance
        """
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)
        return slope
    
    def _calculate_stability(self) -> float:
        """
        Calcule la stabilité de l'optimisation.
        
        Returns:
            Score de stabilité
        """
        if len(self.performance_history) < 2:
            return 0
        
        # Calcul des variations
        performance_variations = np.diff(self.performance_history)
        risk_variations = np.diff(self.risk_history)
        
        # Calcul de la stabilité
        performance_stability = 1 - np.std(performance_variations)
        risk_stability = 1 - np.std(risk_variations)
        
        return (performance_stability + risk_stability) / 2
    
    def get_optimization_status(self) -> Dict[str, str]:
        """
        Obtient le statut de l'optimisation.
        
        Returns:
            Dictionnaire du statut
        """
        if self.last_update is None:
            return {'status': 'not_started'}
        
        time_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        
        if time_since_update < self.config.update_frequency:
            return {
                'status': 'in_progress',
                'time_remaining': f"{self.config.update_frequency - time_since_update:.1f}h"
            }
        else:
            return {'status': 'ready_for_update'}
    
    def reset(self):
        """Réinitialise l'optimiseur."""
        self.last_update = None
        self.performance_history = []
        self.risk_history = []
        self.adaptation_history = []
        self.logger.info("Optimiseur réinitialisé") 