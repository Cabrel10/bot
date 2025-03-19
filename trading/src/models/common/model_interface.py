from typing import Protocol, Dict, Any, Optional, List, Union, runtime_checkable
from pathlib import Path
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd

# Correction du chemin d'importation pour utiliser core.data.data_types au lieu de data.data_types
from ...core.data.data_types import (
    TrainingData,
    ModelPrediction,
    ProcessedData,
    ValidationResult,
    FeatureSet
)

@runtime_checkable
class ModelInterface(Protocol):
    """Interface que tous les modèles de trading doivent implémenter.
    
    Cette interface définit le contrat que chaque modèle doit respecter,
    assurant une interopérabilité entre les différents modèles.
    """

    def train(self, data: pd.DataFrame, **kwargs) -> None:
        """Entraîne le modèle sur les données fournies.
        
        Args:
            data: DataFrame contenant les données d'entraînement
            **kwargs: Paramètres additionnels d'entraînement
        """
        ...

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Génère des prédictions sur les données fournies.
        
        Args:
            data: DataFrame contenant les données de test
            
        Returns:
            np.ndarray: Prédictions du modèle
        """
        ...

    def validate(self, data: ProcessedData) -> ValidationResult:
        """Valide les données d'entrée du modèle.
        
        Args:
            data: Données à valider
            
        Returns:
            Résultat détaillé de la validation
        """
        ...

    def save(self, path: Union[str, Path]) -> None:
        """Sauvegarde le modèle et ses métadonnées.
        
        Args:
            path: Chemin de sauvegarde
        """
        ...

    def load(self, path: Union[str, Path]) -> None:
        """Charge le modèle et ses métadonnées.
        
        Args:
            path: Chemin du modèle à charger
        """
        ...

    def get_features(self) -> FeatureSet:
        """Retourne l'ensemble des features utilisées par le modèle.
        
        Returns:
            Description détaillée des features
        """
        ...

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Retourne les hyperparamètres du modèle.
        
        Returns:
            Dictionnaire des hyperparamètres
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle.
        
        Returns:
            Métadonnées du modèle
        """
        ...

    def is_ready(self) -> bool:
        """Vérifie si le modèle est prêt pour les prédictions.
        
        Returns:
            True si le modèle est entraîné et opérationnel
        """
        ...

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Évalue les performances du modèle.
        
        Args:
            data: DataFrame contenant les données d'évaluation
            
        Returns:
            Dict[str, float]: Métriques de performance
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle.
        
        Returns:
            Dict[str, Any]: Paramètres du modèle
        """
        ...

@runtime_checkable
class TrainableModel(ModelInterface, Protocol):
    """Extension de l'interface pour les modèles qui supportent l'entraînement continu."""

    def update(self, 
               new_data: ProcessedData,
               learning_rate: Optional[float] = None) -> Dict[str, float]:
        """Met à jour le modèle avec de nouvelles données.
        
        Args:
            new_data: Nouvelles données d'entraînement
            learning_rate: Taux d'apprentissage optionnel
            
        Returns:
            Métriques de performance de la mise à jour
        """
        ...

    def get_training_history(self) -> Dict[str, List[float]]:
        """Retourne l'historique d'entraînement.
        
        Returns:
            Historique des métriques d'entraînement
        """
        ...

@runtime_checkable
class EnsembleModel(ModelInterface, Protocol):
    """Extension de l'interface pour les modèles d'ensemble."""

    def add_model(self, model: ModelInterface) -> None:
        """Ajoute un modèle à l'ensemble.
        
        Args:
            model: Modèle à ajouter
        """
        ...

    def remove_model(self, model_name: str) -> None:
        """Retire un modèle de l'ensemble.
        
        Args:
            model_name: Nom du modèle à retirer
        """
        ...

    def get_model_weights(self) -> Dict[str, float]:
        """Retourne les poids des modèles dans l'ensemble.
        
        Returns:
            Dictionnaire des poids par modèle
        """
        ...

class ModelMetrics:
    """Constantes pour les métriques standard des modèles."""
    
    # Métriques de performance
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    ROC_AUC = 'roc_auc'
    
    # Métriques de trading
    SHARPE_RATIO = 'sharpe_ratio'
    MAX_DRAWDOWN = 'max_drawdown'
    WIN_RATE = 'win_rate'
    PROFIT_FACTOR = 'profit_factor'
    
    # Métriques de qualité
    DATA_QUALITY = 'data_quality'
    MODEL_CONFIDENCE = 'model_confidence'
    
    @classmethod
    def get_all_metrics(cls) -> List[str]:
        """Retourne toutes les métriques disponibles."""
        return [attr for attr in dir(cls) 
                if not attr.startswith('_') and isinstance(getattr(cls, attr), str)]

class ModelStatus:
    """États possibles d'un modèle."""
    
    INITIALIZED = 'initialized'
    TRAINING = 'training'
    TRAINED = 'trained'
    VALIDATING = 'validating'
    PREDICTING = 'predicting'
    ERROR = 'error'
    STOPPED = 'stopped'

# Exemple d'utilisation et validation
def validate_model_implementation(model: ModelInterface) -> bool:
    """Valide qu'un modèle implémente correctement l'interface.
    
    Args:
        model: Instance du modèle à valider
        
    Returns:
        True si le modèle est conforme
    """
    try:
        # Vérifie que toutes les méthodes requises sont présentes
        required_methods = [
            'train', 'predict', 'validate', 'save', 'load',
            'get_features', 'get_hyperparameters', 'get_model_info',
            'is_ready', 'evaluate', 'get_parameters'
        ]
        
        for method in required_methods:
            if not hasattr(model, method):
                print(f"Méthode manquante: {method}")
                return False
            
        # Vérifie les signatures des méthodes
        if not isinstance(model, ModelInterface):
            print("Le modèle n'implémente pas correctement l'interface")
            return False
            
        return True
        
    except Exception as e:
        print(f"Erreur lors de la validation: {e}")
        return False

if __name__ == "__main__":
    # Exemple de validation
    from ...models.common.base_model import BaseModel
    
    class TestModel(BaseModel):
        """Modèle de test pour validation."""
        def _default_config(self): return {}
        def _validate_config(self): return True
        def _preprocess_data(self, data): return data
        def _build_model(self): pass
        def _train_impl(self, X, y): return {}
        def _predict_impl(self, X): return np.array([])
    
    # Test de validation
    test_model = TestModel()
    is_valid = validate_model_implementation(test_model)
    print(f"Implémentation valide: {is_valid}")