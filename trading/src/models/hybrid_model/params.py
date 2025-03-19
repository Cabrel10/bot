"""
Paramètres pour le modèle hybride.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class OptimizationParams:
    """Paramètres pour l'optimisation du modèle."""
    
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_iterations_without_improvement: int = 20

@dataclass
class NeuralNetworkParams:
    """Paramètres pour le réseau neuronal."""
    
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    dropout_rate: float = 0.2
    hidden_layers: list = None
    activation: str = 'relu'
    optimizer: str = 'adam'
    loss: str = 'mse'

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32, 16]

@dataclass
class GeneticParams:
    """Paramètres pour l'algorithme génétique."""
    
    chromosome_length: int = 20
    fitness_threshold: float = 0.95
    selection_method: str = 'tournament'
    mutation_type: str = 'gaussian'
    crossover_type: str = 'uniform'
    population_diversity_threshold: float = 0.1
    adaptive_mutation: bool = True
    adaptive_crossover: bool = True

@dataclass
class HybridModelParams:
    """Paramètres du modèle hybride."""
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01
    optimization_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.optimization_params is None:
            self.optimization_params = {
                'population_size': 50,
                'generations': 10,
                'mutation_rate': 0.1
            }
        
        # Paramètres supplémentaires pour le modèle hybride
        self.weight_adjustment_rate: float = 0.1
        self.retraining_interval: int = 100
        self.performance_threshold: float = 0.7
        self.max_model_memory: int = 5
        self.ensemble_method: str = 'weighted_average'
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les paramètres en dictionnaire."""
        return {
            'optimization_params': self.optimization_params,
            'neural_params': asdict(self.neural_params),
            'genetic_params': asdict(self.genetic_params),
            'initial_weights': self.initial_weights,
            'weight_adjustment_rate': self.weight_adjustment_rate,
            'retraining_interval': self.retraining_interval,
            'performance_threshold': self.performance_threshold,
            'max_model_memory': self.max_model_memory,
            'ensemble_method': self.ensemble_method
        }
        
    def save(self, path: Path) -> None:
        """Sauvegarde les paramètres dans un fichier JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def load(self, path: Path) -> None:
        """Charge les paramètres depuis un fichier JSON."""
        with open(path, 'r') as f:
            params = json.load(f)
            
        self.optimization_params = params['optimization_params']
        self.neural_params = NeuralNetworkParams(**params['neural_params'])
        self.genetic_params = GeneticParams(**params['genetic_params'])
        self.initial_weights = params['initial_weights']
        self.weight_adjustment_rate = params['weight_adjustment_rate']
        self.retraining_interval = params['retraining_interval']
        self.performance_threshold = params['performance_threshold']
        self.max_model_memory = params['max_model_memory']
        self.ensemble_method = params['ensemble_method']
        
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'HybridModelParams':
        """Crée une instance à partir d'un dictionnaire."""
        return cls(
            optimization_params=params_dict['optimization_params'],
            neural_params=NeuralNetworkParams(**params_dict['neural_params']),
            genetic_params=GeneticParams(**params_dict['genetic_params']),
            initial_weights=params_dict['initial_weights']
        )
