from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import joblib

from trading.core.data_types import (
    TrainingData, 
    ModelPrediction, 
    ProcessedData, 
    ValidationResult,
    ModelMetrics,
    TrainingConfig,
    PredictionResult
)
from ...utils.logger import TradingLogger
from ...utils.helpers import TradingHelpers

class BaseModel(ABC):
    """Classe abstraite de base pour tous les modèles de trading.
    
    Cette classe définit l'interface commune que tous les modèles
    doivent implémenter, assurant une cohérence dans l'utilisation
    des différents modèles.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialise le modèle avec une configuration optionnelle."""
        self.logger = TradingLogger()
        self.helpers = TradingHelpers()
        self.config = config
        self._is_trained = False
        self._training_history = []
        self._last_training = None
        self.model_info = {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'last_training': None,
            'performance_metrics': {}
        }

    @abstractmethod
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par défaut du modèle."""
        pass

    @abstractmethod
    def _validate_config(self) -> bool:
        """Valide la configuration du modèle."""
        pass

    @abstractmethod
    def _preprocess_data(self, data: ProcessedData) -> TrainingData:
        """Prétraite les données pour l'entraînement ou la prédiction."""
        pass

    @abstractmethod
    def _build_model(self) -> None:
        """Construit l'architecture du modèle."""
        pass

    @abstractmethod
    def _train_impl(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Implémentation spécifique de l'entraînement."""
        pass

    @abstractmethod
    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Implémentation spécifique de la prédiction."""
        pass

    @abstractmethod
    async def train(self, data: TrainingData) -> ModelMetrics:
        """Entraîne le modèle."""
        try:
            # Validation de la configuration
            if not self._validate_config():
                raise ValueError("Configuration invalide")

            # Prétraitement des données
            processed_data = self._preprocess_data(data)
            
            # Construction du modèle si nécessaire
            if not hasattr(self, 'model'):
                self._build_model()

            # Entraînement
            metrics = self._train_impl(processed_data.X, processed_data.y)
            
            # Mise à jour des informations du modèle
            self._is_trained = True
            self.model_info.update({
                'last_training': datetime.now().isoformat(),
                'performance_metrics': metrics,
                'data_info': {
                    'samples': len(processed_data.X),
                    'features': processed_data.feature_names,
                    'timeframe': processed_data.timeframe
                }
            })

            self._training_history.append(metrics)
            self._last_training = metrics

            return metrics

        except Exception as e:
            self.logger.log_error(e, {'action': 'train', 'model': self.model_info['name']})
            raise

    @abstractmethod
    async def predict(self, X: np.ndarray) -> PredictionResult:
        """Génère des prédictions."""
        try:
            if not self._is_trained:
                raise RuntimeError("Le modèle n'est pas entraîné")

            # Prétraitement
            processed_data = self._preprocess_data(X)
            
            # Prédiction
            raw_predictions = self._predict_impl(processed_data.X)
            
            # Conversion en format standardisé
            predictions = []
            for i, pred in enumerate(raw_predictions):
                predictions.append(ModelPrediction(
                    timestamp=processed_data.timestamps[i],
                    symbol=processed_data.symbol,
                    prediction_type=self.config.get('prediction_type', 'direction'),
                    value=float(pred),
                    confidence=self._calculate_confidence(pred),
                    horizon=processed_data.timeframe,
                    model_name=self.model_info['name']
                ))

            return PredictionResult(predictions=predictions)

        except Exception as e:
            self.logger.log_error(e, {'action': 'predict', 'model': self.model_info['name']})
            raise

    @abstractmethod
    async def validate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Valide le modèle sur des données."""
        try:
            processed_data = self._preprocess_data(X)
            
            # Vérifications de base
            missing_values = {
                feature: int(np.isnan(processed_data.X[:, i]).sum())
                for i, feature in enumerate(processed_data.feature_names)
            }
            
            # Détection des valeurs aberrantes (z-score > 3)
            outliers = {}
            for i, feature in enumerate(processed_data.feature_names):
                z_scores = np.abs((processed_data.X[:, i] - np.mean(processed_data.X[:, i])) 
                                / np.std(processed_data.X[:, i]))
                outliers[feature] = list(np.where(z_scores > 3)[0])

            # Score de qualité (0-1)
            data_quality_score = 1.0
            if missing_values:
                data_quality_score -= 0.3
            if any(len(out) > 0 for out in outliers.values()):
                data_quality_score -= 0.2

            return ModelMetrics(
                is_valid=data_quality_score > 0.7,
                missing_values=missing_values,
                outliers=outliers,
                data_quality_score=data_quality_score,
                errors=[]
            )

        except Exception as e:
            self.logger.log_error(e, {'action': 'validate', 'model': self.model_info['name']})
            return ModelMetrics(
                is_valid=False,
                missing_values={},
                outliers={},
                data_quality_score=0.0,
                errors=[str(e)]
            )

    @abstractmethod
    async def save(self, path: str):
        """Sauvegarde le modèle."""
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde du modèle
            model_path = path.with_suffix('.joblib')
            joblib.dump(self.model, model_path)
            
            # Sauvegarde de la configuration et des métadonnées
            config_path = path.with_name(f"{path.stem}_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'config': self.config,
                    'model_info': self.model_info
                }, f, indent=4)

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'save',
                'model': self.model_info['name'],
                'path': str(path)
            })
            raise

    @abstractmethod
    async def load(self, path: str):
        """Charge le modèle."""
        try:
            path = Path(path)
            
            # Chargement du modèle
            model_path = path.with_suffix('.joblib')
            self.model = joblib.load(model_path)
            
            # Chargement de la configuration
            config_path = path.with_name(f"{path.stem}_config.json")
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = data['config']
                self.model_info = data['model_info']
            
            self._is_trained = True

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'load',
                'model': self.model_info['name'],
                'path': str(path)
            })
            raise

    def _calculate_confidence(self, prediction: float) -> float:
        """Calcule le niveau de confiance de la prédiction.
        
        Cette implémentation par défaut peut être surchargée
        par les classes dérivées pour un calcul plus spécifique.
        """
        # Implémentation simple basée sur la distance à 0
        return min(abs(prediction), 1.0)

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle."""
        return self.model_info.copy()

    def reset(self) -> None:
        """Réinitialise le modèle à son état initial."""
        self._is_trained = False
        self.model_info['last_training'] = None
        self.model_info['performance_metrics'] = {}
        if hasattr(self, 'model'):
            delattr(self, 'model')

    def is_trained(self) -> bool:
        """Vérifie si le modèle est entraîné."""
        return self._is_trained

    def get_training_history(self) -> list:
        """Retourne l'historique d'entraînement."""
        return self._training_history

    def get_last_training(self) -> Optional[ModelMetrics]:
        """Retourne les métriques du dernier entraînement."""
        return self._last_training