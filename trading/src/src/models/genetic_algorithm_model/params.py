from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from datetime import datetime

@dataclass
class SelectionParams:
    """Paramètres pour la sélection des individus."""
    method: str  # tournament, roulette, rank
    tournament_size: int
    elitism: int

@dataclass
class CrossoverParams:
    """Paramètres pour le croisement."""
    probability: float
    methods: List[Dict[str, float]]  # [{name: str, weight: float}]

@dataclass
class MutationParams:
    """Paramètres pour la mutation."""
    probability: float
    methods: List[Dict[str, Any]]  # [{name: str, weight: float, params: Dict}]

@dataclass
class FitnessParams:
    """Paramètres pour l'évaluation du fitness."""
    metrics: List[Dict[str, float]]  # [{name: str, weight: float}]
    constraints: Dict[str, Dict[str, float]]

@dataclass
class EvolutionParams:
    """Paramètres pour l'évolution."""
    max_generations: int
    convergence_criteria: Dict[str, float]
    adaptive_rates: Dict[str, Dict[str, float]]

class GeneticAlgorithmParams:
    """Gestionnaire des paramètres de l'algorithme génétique."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise les paramètres depuis un fichier de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config_path = Path(config_path) if config_path else None
        self.params = self._load_config()
        self._validate_params()
        
        # Paramètres dynamiques
        self.current_mutation_rate = self.params['operators']['mutation']['probability']
        self.current_crossover_rate = self.params['operators']['crossover']['probability']
        self.generation_stats: Dict[int, Dict[str, float]] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML."""
        if not self.config_path:
            self.config_path = Path(__file__).parent / 'config.yaml'
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_params(self) -> None:
        """Valide les paramètres chargés."""
        required_sections = ['population', 'operators', 'fitness', 'evolution']
        for section in required_sections:
            if section not in self.params:
                raise ValueError(f"Section manquante dans la configuration: {section}")

    def get_selection_params(self) -> SelectionParams:
        """Retourne les paramètres de sélection."""
        selection_config = self.params['operators']['selection']
        return SelectionParams(
            method=selection_config['method'],
            tournament_size=selection_config['tournament_size'],
            elitism=selection_config['elitism']
        )

    def get_crossover_params(self) -> CrossoverParams:
        """Retourne les paramètres de croisement."""
        crossover_config = self.params['operators']['crossover']
        return CrossoverParams(
            probability=crossover_config['probability'],
            methods=crossover_config['methods']
        )

    def get_mutation_params(self) -> MutationParams:
        """Retourne les paramètres de mutation."""
        mutation_config = self.params['operators']['mutation']
        return MutationParams(
            probability=mutation_config['probability'],
            methods=mutation_config['methods']
        )

    def get_fitness_params(self) -> FitnessParams:
        """Retourne les paramètres de fitness."""
        return FitnessParams(
            metrics=self.params['fitness']['metrics'],
            constraints=self.params['fitness']['constraints']
        )

    def get_evolution_params(self) -> EvolutionParams:
        """Retourne les paramètres d'évolution."""
        evolution_config = self.params['evolution']
        return EvolutionParams(
            max_generations=evolution_config['max_generations'],
            convergence_criteria=evolution_config['convergence_criteria'],
            adaptive_rates=evolution_config['adaptive_rates']
        )

    def update_adaptive_rates(self, 
                            generation: int,
                            improvement: float) -> None:
        """Met à jour les taux adaptatifs selon l'amélioration."""
        if not self.params['evolution']['adaptive_rates']['enabled']:
            return

        adaptive_config = self.params['evolution']['adaptive_rates']
        
        # Ajustement du taux de mutation
        if improvement < self.params['evolution']['convergence_criteria']['min_improvement']:
            # Augmenter le taux de mutation si la population stagne
            self.current_mutation_rate = min(
                self.current_mutation_rate * 1.5,
                adaptive_config['mutation']['max_rate']
            )
        else:
            # Diminuer le taux de mutation si il y a amélioration
            self.current_mutation_rate = max(
                self.current_mutation_rate * 0.9,
                adaptive_config['mutation']['min_rate']
            )

        # Ajustement du taux de croisement
        self.current_crossover_rate = max(
            min(1.0 - self.current_mutation_rate,
                adaptive_config['crossover']['max_rate']),
            adaptive_config['crossover']['min_rate']
        )

    def save_generation_stats(self, 
                            generation: int,
                            stats: Dict[str, float]) -> None:
        """Sauvegarde les statistiques d'une génération."""
        self.generation_stats[generation] = {
            **stats,
            'mutation_rate': self.current_mutation_rate,
            'crossover_rate': self.current_crossover_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarde dans un fichier
        stats_path = self.config_path.parent / 'evolution_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.generation_stats, f, indent=4)

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des paramètres actuels."""
        return {
            'population_size': self.params['population']['size'],
            'current_rates': {
                'mutation': self.current_mutation_rate,
                'crossover': self.current_crossover_rate
            },
            'selection': {
                'method': self.params['operators']['selection']['method'],
                'tournament_size': self.params['operators']['selection']['tournament_size']
            },
            'evolution': {
                'max_generations': self.params['evolution']['max_generations'],
                'convergence_criteria': self.params['evolution']['convergence_criteria']
            }
        }

# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des paramètres
    params = GeneticAlgorithmParams()
    
    # Accès aux différents paramètres
    selection_params = params.get_selection_params()
    crossover_params = params.get_crossover_params()
    mutation_params = params.get_mutation_params()
    fitness_params = params.get_fitness_params()
    
    # Test des taux adaptatifs
    params.update_adaptive_rates(
        generation=1,
        improvement=0.001
    )
    
    # Affichage du résumé
    print("Configuration de l'algorithme génétique:")
    print(json.dumps(params.get_summary(), indent=2))
