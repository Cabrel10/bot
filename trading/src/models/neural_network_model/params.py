from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from datetime import datetime

@dataclass
class LayerParams:
    """Paramètres d'une couche du réseau."""
    type: str
    units: int
    activation: str
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    return_sequences: bool = False
    kernel_regularizer: Optional[Dict[str, Any]] = None

@dataclass
class OptimizerParams:
    """Paramètres de l'optimiseur."""
    type: str
    learning_rate: float
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False

@dataclass
class TrainingParams:
    """Paramètres d'entraînement."""
    batch_size: int
    epochs: int
    validation_split: float
    shuffle: bool
    early_stopping_patience: int
    reduce_lr_patience: int
    min_lr: float

@dataclass
class PreprocessingParams:
    """Paramètres de prétraitement."""
    normalization_method: str
    feature_range: Tuple[float, float]
    sequence_length: int
    padding_type: str
    feature_engineering: Dict[str, bool]

class NeuralNetworkParams:
    """Gestionnaire des paramètres du réseau de neurones."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise les paramètres depuis un fichier de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = Path(config_path) if config_path else None
        self.params = self._load_config()
        self._validate_params()
        
        # Paramètres dynamiques
        self.current_learning_rate = self.params['training']['optimizer']['learning_rate']
        self.current_batch_size = self.params['training']['batch_size']
        self.training_history: Dict[str, List[float]] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML."""
        if not self.config_path:
            self.config_path = Path(__file__).parent / 'config.yaml'
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_params(self) -> None:
        """Valide les paramètres chargés."""
        required_sections = ['model_architecture', 'training', 'preprocessing']
        for section in required_sections:
            if section not in self.params:
                raise ValueError(f"Section manquante dans la configuration: {section}")

    def get_layer_params(self) -> List[LayerParams]:
        """Retourne les paramètres des couches."""
        layers = []
        for layer_config in self.params['model_architecture']['layers']:
            layers.append(LayerParams(**layer_config))
        return layers

    def get_optimizer_params(self) -> OptimizerParams:
        """Retourne les paramètres de l'optimiseur."""
        return OptimizerParams(**self.params['training']['optimizer'])

    def get_training_params(self) -> TrainingParams:
        """Retourne les paramètres d'entraînement."""
        training_config = self.params['training']
        return TrainingParams(
            batch_size=training_config['batch_size'],
            epochs=training_config['epochs'],
            validation_split=training_config['validation_split'],
            shuffle=training_config['shuffle'],
            early_stopping_patience=training_config['callbacks']['early_stopping']['patience'],
            reduce_lr_patience=training_config['callbacks']['reduce_lr']['patience'],
            min_lr=training_config['callbacks']['reduce_lr']['min_lr']
        )

    def get_preprocessing_params(self) -> PreprocessingParams:
        """Retourne les paramètres de prétraitement."""
        preproc_config = self.params['preprocessing']
        return PreprocessingParams(
            normalization_method=preproc_config['normalization']['method'],
            feature_range=tuple(preproc_config['normalization']['feature_range']),
            sequence_length=preproc_config['sequence']['max_length'],
            padding_type=preproc_config['sequence']['padding'],
            feature_engineering=preproc_config['feature_engineering']
        )

    def update_learning_rate(self, new_lr: float) -> None:
        """Met à jour le taux d'apprentissage."""
        self.current_learning_rate = new_lr
        self.params['training']['optimizer']['learning_rate'] = new_lr

    def adjust_batch_size(self, available_memory: int) -> None:
        """Ajuste la taille du batch en fonction de la mémoire disponible."""
        # Formule simple pour ajuster le batch_size
        suggested_batch_size = min(
            self.params['training']['batch_size'],
            available_memory // (1024 * 1024)  # Conversion en MB
        )
        self.current_batch_size = max(1, suggested_batch_size)

    def save_training_history(self, history: Dict[str, List[float]]) -> None:
        """Sauvegarde l'historique d'entraînement."""
        self.training_history = history
        
        # Sauvegarde dans un fichier
        history_path = self.config_path.parent / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'history': history,
                'last_update': datetime.now().isoformat(),
                'current_learning_rate': self.current_learning_rate,
                'current_batch_size': self.current_batch_size
            }, f, indent=4)

    def get_model_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des paramètres du modèle."""
        return {
            'architecture': {
                'type': self.params['model_architecture']['type'],
                'input_dim': self.params['model_architecture']['input_dim'],
                'sequence_length': self.params['model_architecture']['sequence_length'],
                'n_layers': len(self.params['model_architecture']['layers'])
            },
            'training': {
                'optimizer': self.params['training']['optimizer']['type'],
                'learning_rate': self.current_learning_rate,
                'batch_size': self.current_batch_size,
                'epochs': self.params['training']['epochs']
            },
            'preprocessing': {
                'normalization': self.params['preprocessing']['normalization']['method'],
                'sequence_length': self.params['preprocessing']['sequence']['max_length']
            }
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des paramètres
    params = NeuralNetworkParams()
    
    # Accès aux différents paramètres
    layer_params = params.get_layer_params()
    optimizer_params = params.get_optimizer_params()
    training_params = params.get_training_params()
    preprocessing_params = params.get_preprocessing_params()
    
    # Affichage du résumé
    print("Configuration du modèle:")
    print(json.dumps(params.get_model_summary(), indent=2))
    
    # Test d'ajustement des paramètres
    params.update_learning_rate(0.0005)
    params.adjust_batch_size(8000)  # 8GB de mémoire
    
    print("\nParamètres ajustés:")
    print(f"Learning rate: {params.current_learning_rate}")
    print(f"Batch size: {params.current_batch_size}")
