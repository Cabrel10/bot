from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
import concurrent.futures
import json

from .individual import TradingIndividual
from .params import GeneticAlgorithmParams
from ...data.data_types import ProcessedData
from ...utils.logger import TradingLogger

class Population:
    """Gère une population d'individus pour l'algorithme génétique."""

    def __init__(self, params: Optional[GeneticAlgorithmParams] = None):
        """Initialise la population.
        
        Args:
            params: Paramètres de l'algorithme génétique
        """
        self.params = params or GeneticAlgorithmParams()
        self.individuals: List[TradingIndividual] = []
        self.generation: int = 0
        self.best_individual: Optional[TradingIndividual] = None
        self.logger = TradingLogger()
        self.stats: Dict[str, List[float]] = {
            'avg_fitness': [],
            'best_fitness': [],
            'worst_fitness': [],
            'diversity': []
        }

    async def initialize(self, size: Optional[int] = None) -> None:
        """Initialise la population avec des individus aléatoires.
        
        Args:
            size: Taille de la population (utilise la config par défaut si None)
        """
        try:
            population_size = size or self.params.params['population']['size']
            self.individuals = [
                TradingIndividual() for _ in range(population_size)
            ]
            self.generation = 0
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'initialize_population'})
            raise

    async def evaluate(self, data: ProcessedData) -> None:
        """Évalue tous les individus de la population.
        
        Args:
            data: Données pour l'évaluation
        """
        try:
            # Évaluation parallèle si configurée
            if self.params.params.get('parallel_processing', {}).get('enabled', False):
                await self._parallel_evaluation(data)
            else:
                await self._sequential_evaluation(data)
            
            # Tri par fitness
            self.individuals.sort(key=lambda x: x.stats.fitness_score, reverse=True)
            
            # Mise à jour du meilleur individu
            if not self.best_individual or \
               self.individuals[0].stats.fitness_score > self.best_individual.stats.fitness_score:
                self.best_individual = self.individuals[0].clone()
            
            # Mise à jour des statistiques
            self._update_stats()
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'evaluate_population'})
            raise

    async def _parallel_evaluation(self, data: ProcessedData) -> None:
        """Évalue la population en parallèle."""
        max_workers = self.params.params['parallel_processing']['max_workers']
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                individual.evaluate(data)
                for individual in self.individuals
            ]
            await asyncio.gather(*tasks)

    async def _sequential_evaluation(self, data: ProcessedData) -> None:
        """Évalue la population séquentiellement."""
        for individual in self.individuals:
            await individual.evaluate(data)

    def _update_stats(self) -> None:
        """Met à jour les statistiques de la population."""
        try:
            fitness_scores = [ind.stats.fitness_score for ind in self.individuals]
            
            self.stats['avg_fitness'].append(np.mean(fitness_scores))
            self.stats['best_fitness'].append(max(fitness_scores))
            self.stats['worst_fitness'].append(min(fitness_scores))
            self.stats['diversity'].append(self._calculate_diversity())
            
            # Sauvegarde des stats
            self._save_stats()
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'update_stats'})

    def _calculate_diversity(self) -> float:
        """Calcule la diversité de la population."""
        try:
            # Moyenne des distances entre les chromosomes
            distances = []
            for i in range(len(self.individuals)):
                for j in range(i + 1, len(self.individuals)):
                    dist = np.mean([
                        np.linalg.norm(
                            self.individuals[i].chromosome.segments[key].values -
                            self.individuals[j].chromosome.segments[key].values
                        )
                        for key in self.individuals[i].chromosome.segments
                    ])
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'calculate_diversity'})
            return 0.0

    async def evolve(self) -> None:
        """Fait évoluer la population vers la génération suivante."""
        try:
            new_population: List[TradingIndividual] = []
            
            # Élitisme
            selection_params = self.params.get_selection_params()
            elite_size = selection_params.elitism
            new_population.extend([ind.clone() for ind in self.individuals[:elite_size]])
            
            # Reproduction
            while len(new_population) < len(self.individuals):
                # Sélection des parents
                parent1 = self._select_parent(selection_params)
                parent2 = self._select_parent(selection_params)
                
                # Croisement
                if np.random.random() < self.params.current_crossover_rate:
                    children = parent1.crossover(parent2)
                    new_population.extend(children)
                else:
                    new_population.extend([parent1.clone(), parent2.clone()])
            
            # Mutation
            for individual in new_population[elite_size:]:  # Pas de mutation des élites
                if np.random.random() < self.params.current_mutation_rate:
                    mutation_params = self.params.get_mutation_params()
                    method = np.random.choice(
                        [m['name'] for m in mutation_params.methods],
                        p=[m['weight'] for m in mutation_params.methods]
                    )
                    individual.mutate(self.params.current_mutation_rate, method)
            
            # Mise à jour de la population
            self.individuals = new_population[:len(self.individuals)]
            self.generation += 1
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'evolve_population'})
            raise

    def _select_parent(self, selection_params: Any) -> TradingIndividual:
        """Sélectionne un parent pour la reproduction."""
        method = selection_params.method
        
        if method == 'tournament':
            # Sélection par tournoi
            tournament = np.random.choice(
                self.individuals,
                size=selection_params.tournament_size,
                replace=False
            )
            return max(tournament, key=lambda x: x.stats.fitness_score)
            
        elif method == 'roulette':
            # Sélection par roulette
            fitness_sum = sum(ind.stats.fitness_score for ind in self.individuals)
            if fitness_sum <= 0:
                return np.random.choice(self.individuals)
            
            probs = [ind.stats.fitness_score/fitness_sum for ind in self.individuals]
            return np.random.choice(self.individuals, p=probs)
            
        else:
            # Sélection aléatoire par défaut
            return np.random.choice(self.individuals)

    def _save_stats(self) -> None:
        """Sauvegarde les statistiques de la population."""
        try:
            stats_path = Path(self.params.config_path).parent / 'population_stats.json'
            with open(stats_path, 'w') as f:
                json.dump({
                    'generation': self.generation,
                    'timestamp': datetime.now().isoformat(),
                    'population_size': len(self.individuals),
                    'best_individual': self.best_individual.get_summary() if self.best_individual else None,
                    'stats': self.stats,
                    'current_rates': {
                        'mutation': self.params.current_mutation_rate,
                        'crossover': self.params.current_crossover_rate
                    }
                }, f, indent=4)
                
        except Exception as e:
            self.logger.log_error(e, {'action': 'save_stats'})

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de la population."""
        return {
            'generation': self.generation,
            'size': len(self.individuals),
            'best_fitness': self.stats['best_fitness'][-1] if self.stats['best_fitness'] else None,
            'avg_fitness': self.stats['avg_fitness'][-1] if self.stats['avg_fitness'] else None,
            'diversity': self.stats['diversity'][-1] if self.stats['diversity'] else None,
            'best_individual': self.best_individual.get_summary() if self.best_individual else None
        }

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création de la population
        population = Population()
        
        # Initialisation
        await population.initialize(size=50)
        
        # Évaluation
        data = ProcessedData(...)  # À compléter
        await population.evaluate(data)
        
        # Évolution
        await population.evolve()
        
        # Affichage des résultats
        print("Résumé de la population:")
        print(json.dumps(population.get_summary(), indent=2))

    # Exécution
    asyncio.run(main())