"""
Gestionnaire d'entraînement pour les modèles de trading.
"""
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
from pathlib import Path
import json

from ...core.data_types import (
    TrainingData,
    ProcessedData,
    ModelMetrics,
    ValidationResult
)
from ...models.hybrid_model import HybridModel, HybridModelParams
from ...utils.logger import TradingLogger

@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement."""
    model_type: str = "hybrid"  # "hybrid", "neural", "genetic"
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    use_gpu: bool = True
    random_seed: int = 42
    data_config: Dict[str, Any] = None
    model_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.data_config is None:
            self.data_config = {}
        if self.model_config is None:
            self.model_config = {}

class TrainingManager:
    """Gestionnaire pour l'entraînement des modèles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialise le gestionnaire d'entraînement.
        
        Args:
            config: Configuration optionnelle
        """
        self.config = TrainingConfig(**config if config else {})
        self.logger = TradingLogger()
        self.model = None
        self.training_status = {
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'best_metrics': None,
            'start_time': None,
            'end_time': None
        }
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Initialise les répertoires nécessaires."""
        try:
            for directory in [
                self.config.checkpoint_dir,
                self.config.model_save_dir,
                self.config.log_dir
            ]:
                Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des répertoires: {e}")
            raise

    async def start_training(self,
                           training_data: ProcessedData,
                           validation_data: Optional[ProcessedData] = None) -> None:
        """Démarre l'entraînement du modèle.
        
        Args:
            training_data: Données d'entraînement
            validation_data: Données de validation (optionnel)
        """
        try:
            if self.training_status['is_training']:
                raise RuntimeError("Un entraînement est déjà en cours")

            self.training_status.update({
                'is_training': True,
                'current_epoch': 0,
                'total_epochs': self.config.epochs,
                'start_time': datetime.now(),
                'best_metrics': None
            })

            # Initialisation du modèle
            if self.model is None:
                self.model = self._create_model()

            # Entraînement
            metrics = await self.model.train(
                training_data=training_data,
                validation_data=validation_data
            )

            # Mise à jour du statut
            self.training_status.update({
                'is_training': False,
                'end_time': datetime.now(),
                'best_metrics': metrics
            })

            # Sauvegarde du modèle
            await self._save_model()

        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}")
            self.training_status['is_training'] = False
            raise

    def _create_model(self) -> Any:
        """Crée une instance du modèle selon la configuration."""
        if self.config.model_type == "hybrid":
            return HybridModel(self.config.model_config)
        else:
            raise ValueError(f"Type de modèle non supporté: {self.config.model_type}")

    async def stop_training(self) -> None:
        """Arrête l'entraînement en cours."""
        if self.training_status['is_training']:
            # Implémentation de l'arrêt
            self.training_status['is_training'] = False
            self.logger.info("Entraînement arrêté")

    async def _save_model(self) -> None:
        """Sauvegarde le modèle entraîné."""
        try:
            if self.model is None:
                raise ValueError("Aucun modèle à sauvegarder")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = Path(self.config.model_save_dir) / f"model_{timestamp}.pkl"
            
            # Sauvegarde du modèle
            await self.model.save(model_path)
            
            # Sauvegarde des métadonnées
            metadata = {
                'timestamp': timestamp,
                'config': self.config.__dict__,
                'metrics': self.training_status['best_metrics'],
                'training_duration': (
                    self.training_status['end_time'] - 
                    self.training_status['start_time']
                ).total_seconds() if self.training_status['end_time'] else None
            }
            
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise

    def get_training_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel de l'entraînement."""
        return self.training_status

    async def validate_model(self,
                           validation_data: ProcessedData) -> ValidationResult:
        """Valide le modèle sur un ensemble de données.
        
        Args:
            validation_data: Données de validation
            
        Returns:
            ValidationResult: Résultats de la validation
        """
        try:
            if self.model is None:
                raise ValueError("Aucun modèle à valider")

            # Validation
            metrics = await self.model.validate(validation_data)
            
            return ValidationResult(
                is_valid=True,
                missing_values={},  # À implémenter
                outliers={},        # À implémenter
                data_quality_score=metrics.get('validation_score', 0.0),
                errors=[]
            )

        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {e}")
            return ValidationResult(
                is_valid=False,
                missing_values={},
                outliers={},
                data_quality_score=0.0,
                errors=[str(e)]
            )

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Charge un modèle sauvegardé.
        
        Args:
            model_path: Chemin vers le modèle
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

            # Chargement des métadonnées
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.config = TrainingConfig(**metadata['config'])

            # Création et chargement du modèle
            self.model = self._create_model()
            self.model.load(model_path)

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

    async def update_model(self,
                         new_data: ProcessedData,
                         update_config: Optional[Dict[str, Any]] = None) -> None:
        """Met à jour le modèle avec de nouvelles données.
        
        Args:
            new_data: Nouvelles données d'entraînement
            update_config: Configuration pour la mise à jour
        """
        try:
            if self.model is None:
                raise ValueError("Aucun modèle à mettre à jour")

            if update_config:
                # Mise à jour de la configuration
                self.config = TrainingConfig(**{
                    **self.config.__dict__,
                    **update_config
                })

            # Mise à jour du modèle
            await self.model.update(new_data)
            
            # Sauvegarde du modèle mis à jour
            await self._save_model()

        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du modèle: {e}")
            raise 