import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import random
import mlflow
from concurrent.futures import ProcessPoolExecutor

@dataclass
class Individual:
    """Represents a trading strategy individual."""
    parameters: Dict[str, float]
    fitness: float = float('-inf')

class GeneticOptimizer:
    """Genetic algorithm for trading strategy optimization."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the genetic optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self._setup_mlflow()

    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 5,
            'tournament_size': 3,
            'parameter_bounds': {
                'entry_threshold': (0.1, 0.9),
                'exit_threshold': (0.1, 0.9),
                'stop_loss': (0.01, 0.05),
                'take_profit': (0.02, 0.1),
                'lookback_period': (10, 100),
                'risk_ratio': (0.01, 0.05)
            }
        }

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        mlflow.set_experiment('genetic_optimization')

    def initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.config['population_size']):
            parameters = {}
            for param, (min_val, max_val) in self.config['parameter_bounds'].items():
                parameters[param] = random.uniform(min_val, max_val)
            self.population.append(Individual(parameters=parameters))

    def evaluate_population(self, fitness_function: Callable[[Dict], float]) -> None:
        """Evaluate fitness for all individuals.

        Args:
            fitness_function: Function to evaluate individual fitness
        """
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(fitness_function, ind.parameters)
                      for ind in self.population]
            
            for ind, future in zip(self.population, futures):
                ind.fitness = future.result()

        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best

    def _tournament_selection(self) -> Individual:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, self.config['tournament_size'])
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.config['crossover_rate']:
            return parent1, parent2

        child1_params = {}
        child2_params = {}

        for param in self.config['parameter_bounds'].keys():
            if random.random() < 0.5:
                child1_params[param] = parent1.parameters[param]
                child2_params[param] = parent2.parameters[param]
            else:
                child1_params[param] = parent2.parameters[param]
                child2_params[param] = parent1.parameters[param]

        return Individual(parameters=child1_params), Individual(parameters=child2_params)

    def _mutate(self, individual: Individual) -> None:
        """Mutate an individual."""
        for param, (min_val, max_val) in self.config['parameter_bounds'].items():
            if random.random() < self.config['mutation_rate']:
                delta = (max_val - min_val) * 0.1
                value = individual.parameters[param] + random.uniform(-delta, delta)
                individual.parameters[param] = max(min_val, min(max_val, value))

    def evolve(self, fitness_function: Callable[[Dict], float]) -> None:
        """Evolve population for one generation.

        Args:
            fitness_function: Function to evaluate individual fitness
        """
        with mlflow.start_run(run_name=f'generation_{self.generation}'):
            # Evaluate current population
            self.evaluate_population(fitness_function)

            # Log metrics
            best_fitness = max(ind.fitness for ind in self.population)
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            mlflow.log_metrics({
                'best_fitness': best_fitness,
                'average_fitness': avg_fitness
            })

            # Elitism
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config['elite_size']]
            new_population = elite.copy()

            # Generate new individuals
            while len(new_population) < self.config['population_size']:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)

                self._mutate(child1)
                self._mutate(child2)

                new_population.extend([child1, child2])

            self.population = new_population[:self.config['population_size']]
            self.generation += 1

    def run_optimization(self, fitness_function: Callable[[Dict], float]) -> Dict:
        """Run complete genetic optimization.

        Args:
            fitness_function: Function to evaluate individual fitness

        Returns:
            Best parameters found
        """
        self.initialize_population()

        for generation in range(self.config['generations']):
            self.evolve(fitness_function)

        return {
            'best_parameters': self.best_individual.parameters,
            'best_fitness': self.best_individual.fitness,
            'generations': self.generation
        }