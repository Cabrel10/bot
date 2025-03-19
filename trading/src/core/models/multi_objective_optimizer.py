"""
Optimiseur multi-objectif pour les stratégies de trading.
Implémente NSGA-II (Non-dominated Sorting Genetic Algorithm II).
"""
from typing import List, Dict, Tuple, Optional, Callable, Union, Any, NamedTuple
import numpy as np
from dataclasses import dataclass, field
import random
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from numba import jit
from sklearn.model_selection import TimeSeriesSplit

from trading.exceptions import (
    ValidationError,
    OptimizationError,
    ConfigurationError,
    DataError
)

@dataclass
class FuturesConfig:
    """
    Configuration pour les contrats futures.
    
    Attributes:
        enabled (bool): Active le support des futures
        margin_requirement (float): Exigence de marge [0,1]
        contract_size (float): Taille du contrat
        commission_rate (float): Taux de commission
        slippage (float): Glissement moyen par trade
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    enabled: bool = False
    margin_requirement: float = 0.1
    contract_size: float = 1.0
    commission_rate: float = 0.0001
    slippage: float = 0.0001
    
    def __post_init__(self):
        """Valide les paramètres après l'initialisation."""
        if self.enabled:
            if not 0 < self.margin_requirement <= 1:
                raise ValidationError(
                    f"L'exigence de marge doit être entre 0 et 1, reçu: {self.margin_requirement}"
                )
            if self.contract_size <= 0:
                raise ValidationError(
                    f"La taille du contrat doit être > 0, reçu: {self.contract_size}"
                )
            if self.commission_rate < 0:
                raise ValidationError(
                    f"Le taux de commission doit être >= 0, reçu: {self.commission_rate}"
                )
            if self.slippage < 0:
                raise ValidationError(
                    f"Le glissement doit être >= 0, reçu: {self.slippage}"
                )

@dataclass
class VolumeConfig:
    """
    Configuration pour les indicateurs de volume.
    
    Attributes:
        enabled (bool): Active les indicateurs de volume
        min_volume (float): Volume minimum pour trader
        volume_ma_periods (List[int]): Périodes pour les moyennes mobiles
        volume_impact (float): Impact du volume sur le signal [0,1]
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    enabled: bool = False
    min_volume: float = 1000.0
    volume_ma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    volume_impact: float = 0.3
    
    def __post_init__(self):
        """Valide les paramètres après l'initialisation."""
        if self.enabled:
            if self.min_volume <= 0:
                raise ValidationError(
                    f"Le volume minimum doit être > 0, reçu: {self.min_volume}"
                )
            if not self.volume_ma_periods:
                raise ValidationError("Au moins une période MA doit être définie")
            if not all(p > 0 for p in self.volume_ma_periods):
                raise ValidationError("Toutes les périodes doivent être > 0")
            if not 0 <= self.volume_impact <= 1:
                raise ValidationError(
                    f"L'impact du volume doit être entre 0 et 1, reçu: {self.volume_impact}"
                )

@dataclass
class TimeSeriesValidationConfig:
    """
    Configuration pour la validation croisée temporelle.
    
    Attributes:
        enabled (bool): Active la validation croisée temporelle
        n_splits (int): Nombre de splits temporels
        train_size (float): Proportion des données d'entraînement [0,1]
        gap (int): Nombre de périodes entre train et test
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    enabled: bool = False
    n_splits: int = 5
    train_size: float = 0.8
    gap: int = 0
    
    def __post_init__(self):
        """Valide les paramètres après l'initialisation."""
        if self.enabled:
            if self.n_splits < 2:
                raise ValidationError(
                    f"Le nombre de splits doit être >= 2, reçu: {self.n_splits}"
                )
            if not 0 < self.train_size < 1:
                raise ValidationError(
                    f"La taille d'entraînement doit être entre 0 et 1, reçu: {self.train_size}"
                )
            if self.gap < 0:
                raise ValidationError(
                    f"Le gap doit être >= 0, reçu: {self.gap}"
                )

@dataclass
class ObjectiveConfig:
    """Configuration d'un objectif d'optimisation."""
    name: str
    weight: float
    minimize: bool = False
    constraint: Optional[float] = None

@dataclass
class NSGAConfig:
    """
    Configuration de l'algorithme NSGA-II.
    
    Attributes:
        population_size (int): Taille de la population
        generations (int): Nombre maximum de générations
        mutation_rate (float): Taux initial de mutation [0,1]
        crossover_rate (float): Taux de croisement [0,1]
        tournament_size (int): Taille du tournoi pour la sélection
        objectives (List[ObjectiveConfig]): Liste des objectifs à optimiser
        parallel_evaluation (bool): Active l'évaluation parallèle
        adaptive_mutation (bool): Active l'adaptation du taux de mutation
        min_mutation_rate (float): Taux minimum de mutation
        max_mutation_rate (float): Taux maximum de mutation
        convergence_threshold (float): Seuil de convergence
        max_stagnation (int): Nombre max de générations sans amélioration
        futures_config (FuturesConfig): Configuration des futures
        volume_config (VolumeConfig): Configuration des volumes
        time_series_validation (TimeSeriesValidationConfig): Configuration de la validation
        
    Raises:
        ValidationError: Si les paramètres sont invalides
        ConfigurationError: Si la configuration est incohérente
    """
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    objectives: List[ObjectiveConfig] = field(default_factory=list)
    parallel_evaluation: bool = True
    adaptive_mutation: bool = True
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.4
    convergence_threshold: float = 1e-6
    max_stagnation: int = 15
    futures_config: FuturesConfig = field(default_factory=FuturesConfig)
    volume_config: VolumeConfig = field(default_factory=VolumeConfig)
    time_series_validation: TimeSeriesValidationConfig = field(
        default_factory=TimeSeriesValidationConfig
    )

@dataclass
class Solution:
    """
    Représente une solution dans l'espace multi-objectif.
    
    Attributes:
        parameters (Dict[str, float]): Paramètres de la solution
        objectives (Dict[str, float]): Valeurs des objectifs
        crowding_distance (float): Distance de crowding
        rank (int): Rang dans le tri non-dominé
        dominated_solutions (List[Solution]): Solutions dominées
        domination_count (int): Nombre de solutions qui dominent celle-ci
        metadata (Dict[str, Any]): Métadonnées additionnelles
        futures_metrics (Dict[str, float]): Métriques spécifiques aux futures
        volume_metrics (Dict[str, float]): Métriques liées au volume
        validation_scores (List[float]): Scores de validation croisée
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    parameters: Dict[str, float]
    objectives: Dict[str, float] = field(default_factory=dict)
    crowding_distance: float = 0.0
    rank: int = 0
    dominated_solutions: List['Solution'] = field(default_factory=list)
    domination_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    futures_metrics: Dict[str, float] = field(default_factory=dict)
    volume_metrics: Dict[str, float] = field(default_factory=dict)
    validation_scores: List[float] = field(default_factory=list)

@jit(nopython=True)
def _fast_non_dominated_sort(objectives_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Version optimisée du tri non-dominé avec Numba.
    
    Args:
        objectives_array: Tableau numpy des valeurs d'objectifs (n_solutions x n_objectives)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Rangs et compteurs de domination
    """
    n_solutions = objectives_array.shape[0]
    domination_counts = np.zeros(n_solutions, dtype=np.int32)
    ranks = np.zeros(n_solutions, dtype=np.int32)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i != j:
                if np.all(objectives_array[i] <= objectives_array[j]) and \
                   np.any(objectives_array[i] < objectives_array[j]):
                    domination_counts[j] += 1
                    
    current_rank = 0
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            break
            
        ranks[current_front] = current_rank
        for i in current_front:
            for j in range(n_solutions):
                if i != j and np.all(objectives_array[i] <= objectives_array[j]):
                    domination_counts[j] -= 1
                    
        domination_counts[current_front] = -1
        current_rank += 1
        
    return ranks, domination_counts

@jit(nopython=True)
def _fast_crowding_distance(objectives_array: np.ndarray, front_indices: np.ndarray) -> np.ndarray:
    """
    Version optimisée du calcul de la distance de crowding avec Numba.
    
    Args:
        objectives_array: Tableau numpy des valeurs d'objectifs
        front_indices: Indices des solutions dans le front
        
    Returns:
        np.ndarray: Distances de crowding
    """
    n_solutions = len(front_indices)
    n_objectives = objectives_array.shape[1]
    distances = np.zeros(n_solutions)
    
    if n_solutions <= 2:
        distances[:] = np.inf
        return distances
        
    for obj in range(n_objectives):
        obj_values = objectives_array[front_indices, obj]
        sorted_indices = np.argsort(obj_values)
        obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        
        if obj_range > 0:
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            for i in range(1, n_solutions - 1):
                distances[sorted_indices[i]] += (
                    obj_values[sorted_indices[i + 1]] - 
                    obj_values[sorted_indices[i - 1]]
                ) / obj_range
                
    return distances

class NSGAII:
    """
    Implémentation de NSGA-II pour l'optimisation multi-objectif
    des stratégies de trading.
    
    Cette classe implémente l'algorithme NSGA-II avec:
    - Tri non-dominé rapide optimisé avec Numba
    - Calcul de la distance de crowding optimisé
    - Sélection par tournoi
    - Croisement et mutation adaptative
    - Support pour les contraintes et les futures
    - Évaluation parallèle
    - Validation croisée temporelle
    """
    
    def __init__(self, config: NSGAConfig):
        """
        Initialise l'algorithme NSGA-II.
        
        Args:
            config: Configuration de l'algorithme
            
        Raises:
            ValidationError: Si la configuration est invalide
        """
        if not isinstance(config, NSGAConfig):
            raise ValidationError("config doit être une instance de NSGAConfig")
            
        self.config = config
        self.population: List[Solution] = []
        self.fronts: List[List[Solution]] = []
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.generation = 0
        self.history: List[Dict] = []
        self.best_solutions: List[Solution] = []
        self.time_series_splits = None
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_time_series_validation(self, data_length: int) -> None:
        """
        Configure la validation croisée temporelle.
        
        Args:
            data_length: Longueur des données
            
        Raises:
            ValidationError: Si la configuration est invalide
        """
        if not self.config.time_series_validation.enabled:
            return
            
        try:
            tscv = TimeSeriesSplit(
                n_splits=self.config.time_series_validation.n_splits,
                gap=self.config.time_series_validation.gap
            )
            
            train_size = int(data_length * self.config.time_series_validation.train_size)
            self.time_series_splits = list(tscv.split(np.arange(data_length)))
            
            self.logger.info(
                f"Validation croisée temporelle configurée avec {len(self.time_series_splits)} splits"
            )
            
        except Exception as e:
            raise ValidationError(f"Erreur de configuration de la validation: {str(e)}")
            
    def evaluate_solution(self, 
                         solution: Solution,
                         evaluation_function: Callable,
                         data: np.ndarray) -> None:
        """
        Évalue une solution avec validation croisée si activée.
        
        Args:
            solution: Solution à évaluer
            evaluation_function: Fonction d'évaluation
            data: Données d'entraînement
            
        Raises:
            OptimizationError: Si l'évaluation échoue
        """
        try:
            if self.config.time_series_validation.enabled:
                scores = []
                for train_idx, val_idx in self.time_series_splits:
                    train_data = data[train_idx]
                    val_data = data[val_idx]
                    
                    # Évaluation sur l'ensemble d'entraînement
                    train_objectives = evaluation_function(
                        solution.parameters,
                        train_data
                    )
                    
                    # Évaluation sur l'ensemble de validation
                    val_objectives = evaluation_function(
                        solution.parameters,
                        val_data
                    )
                    
                    # Calcul du score de validation
                    score = sum(
                        obj.weight * val_objectives[obj.name]
                        for obj in self.config.objectives
                    )
                    scores.append(score)
                    
                solution.validation_scores = scores
                solution.objectives = train_objectives  # Utilise les objectifs d'entraînement
                
            else:
                # Évaluation standard sans validation croisée
                solution.objectives = evaluation_function(
                    solution.parameters,
                    data
                )
                
        except Exception as e:
            raise OptimizationError(f"Erreur lors de l'évaluation: {str(e)}")
            
    def evaluate_population(self, 
                          evaluation_function: Callable,
                          data: np.ndarray) -> None:
        """
        Évalue toute la population avec support pour l'évaluation parallèle.
        
        Args:
            evaluation_function: Fonction d'évaluation
            data: Données d'entraînement
            
        Raises:
            OptimizationError: Si l'évaluation échoue
        """
        try:
            if self.config.parallel_evaluation:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            self.evaluate_solution,
                            solution,
                            evaluation_function,
                            data
                        )
                        for solution in self.population
                    ]
                    for future in futures:
                        future.result()
            else:
                for solution in self.population:
                    self.evaluate_solution(
                        solution,
                        evaluation_function,
                        data
                    )
                    
        except Exception as e:
            raise OptimizationError(f"Erreur lors de l'évaluation: {str(e)}")
            
    def non_dominated_sort(self) -> None:
        """
        Effectue le tri non-dominé rapide avec Numba.
        
        Raises:
            OptimizationError: Si le tri échoue
        """
        try:
            # Conversion en tableau numpy pour Numba
            objectives_list = []
            for solution in self.population:
                obj_values = [
                    solution.objectives[obj.name]
                    for obj in self.config.objectives
                ]
                objectives_list.append(obj_values)
                
            objectives_array = np.array(objectives_list)
            
            # Tri rapide avec Numba
            ranks, domination_counts = _fast_non_dominated_sort(objectives_array)
            
            # Mise à jour des solutions
            self.fronts = []
            current_front = []
            current_rank = 0
            
            while True:
                current_indices = np.where(ranks == current_rank)[0]
                if len(current_indices) == 0:
                    break
                    
                current_front = [
                    self.population[i] for i in current_indices
                ]
                
                # Calcul des distances de crowding
                front_objectives = objectives_array[current_indices]
                distances = _fast_crowding_distance(
                    objectives_array,
                    current_indices
                )
                
                for solution, distance in zip(current_front, distances):
                    solution.rank = current_rank
                    solution.crowding_distance = distance
                    
                self.fronts.append(current_front)
                current_rank += 1
                
        except Exception as e:
            raise OptimizationError(f"Erreur lors du tri non-dominé: {str(e)}")
            
    def optimize(self,
                evaluation_function: Callable,
                data: np.ndarray,
                parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Solution]:
        """
        Optimise la population avec validation croisée temporelle si activée.
        
        Args:
            evaluation_function: Fonction d'évaluation
            data: Données d'entraînement
            parameter_bounds: Bornes des paramètres
            
        Returns:
            List[Solution]: Meilleures solutions trouvées
            
        Raises:
            OptimizationError: Si l'optimisation échoue
        """
        try:
            self.parameter_bounds = parameter_bounds
            
            # Configuration de la validation croisée
            if self.config.time_series_validation.enabled:
                self._setup_time_series_validation(len(data))
                
            # Initialisation de la population
            self.initialize_population(parameter_bounds)
            
            start_time = datetime.now()
            best_hypervolume = float('-inf')
            stagnation_counter = 0
            
            for generation in range(self.config.generations):
                self.generation = generation
                
                # Évaluation
                self.evaluate_population(evaluation_function, data)
                
                # Tri non-dominé
                self.non_dominated_sort()
                
                # Calcul de l'hypervolume
                current_hypervolume = self._calculate_hypervolume()
                
                if current_hypervolume > best_hypervolume:
                    best_hypervolume = current_hypervolume
                    self.best_solutions = self.fronts[0].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
                # Vérification de la convergence
                if stagnation_counter >= self.config.max_stagnation:
                    self.logger.info(
                        f"Convergence atteinte après {generation} générations"
                    )
                    break
                    
                # Création de la nouvelle génération
                self.create_next_generation()
                
                # Adaptation des taux
                if self.config.adaptive_mutation:
                    self._adapt_rates()
                    
                # Enregistrement de l'historique
                self._record_history()
                
                # Log de progression
                if generation % 10 == 0:
                    self._log_progress()
                    
            execution_time = datetime.now() - start_time
            self.logger.info(f"Optimisation terminée en {execution_time}")
            
            return self.best_solutions
            
        except Exception as e:
            raise OptimizationError(f"Erreur lors de l'optimisation: {str(e)}")
            
    @jit(nopython=True)
    def _calculate_hypervolume(self) -> float:
        """
        Calcule l'hypervolume du front de Pareto (version optimisée).
        
        Returns:
            float: Valeur de l'hypervolume
        """
        if not self.fronts:
            return float('-inf')
            
        front = self.fronts[0]
        if not front:
            return float('-inf')
            
        objectives = np.array([
            [solution.objectives[obj.name] for obj in self.config.objectives]
            for solution in front
        ])
        
        reference_point = np.max(objectives, axis=0) * 1.1
        
        return self._compute_hypervolume(objectives, reference_point)
        
    @staticmethod
    @jit(nopython=True)
    def _compute_hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
        """
        Calcule l'hypervolume par la méthode de Monte Carlo (optimisée).
        
        Args:
            points: Points du front de Pareto
            reference: Point de référence
            
        Returns:
            float: Estimation de l'hypervolume
        """
        n_samples = 10000
        n_objectives = points.shape[1]
        
        samples = np.random.uniform(
            low=np.min(points, axis=0),
            high=reference,
            size=(n_samples, n_objectives)
        )
        
        dominated_count = 0
        for sample in samples:
            for point in points:
                if np.all(point <= sample):
                    dominated_count += 1
                    break
                    
        volume = np.prod(reference - np.min(points, axis=0))
        return (dominated_count / n_samples) * volume

    def initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """Initialise la population avec des solutions aléatoires."""
        self.parameter_bounds = parameter_bounds
        self.population = []
        
        for _ in range(self.config.population_size):
            parameters = {
                param: random.uniform(bounds[0], bounds[1])
                for param, bounds in parameter_bounds.items()
            }
            self.population.append(Solution(parameters))

    def create_next_generation(self):
        """Crée la prochaine génération de solutions."""
        offspring = []
        
        while len(offspring) < self.config.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # Combine parents and offspring
        self.population.extend(offspring[:self.config.population_size])
        self.non_dominated_sort()
        self.calculate_crowding_distance()
        
        # Select the best solutions for the next generation
        next_generation = []
        front_idx = 0
        
        while len(next_generation) + len(self.fronts[front_idx]) <= self.config.population_size:
            next_generation.extend(self.fronts[front_idx])
            front_idx += 1
        
        if len(next_generation) < self.config.population_size:
            self.fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            next_generation.extend(
                self.fronts[front_idx][:self.config.population_size - len(next_generation)]
            )
        
        self.population = next_generation

    def tournament_selection(self) -> Solution:
        """Sélectionne une solution par tournoi."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: (x.rank, x.crowding_distance))

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Effectue un croisement entre deux parents."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        child1_params = {}
        child2_params = {}
        
        for param in self.parameter_bounds.keys():
            if random.random() < 0.5:
                child1_params[param] = parent1.parameters[param]
                child2_params[param] = parent2.parameters[param]
            else:
                child1_params[param] = parent2.parameters[param]
                child2_params[param] = parent1.parameters[param]
        
        return Solution(child1_params), Solution(child2_params)

    def mutate(self, solution: Solution):
        """Applique une mutation à une solution."""
        if not self.config.adaptive_mutation:
            mutation_rate = self.config.mutation_rate
        else:
            # Mutation rate adaptative basée sur le rang
            mutation_rate = self.config.mutation_rate * (1 + solution.rank / len(self.fronts))
        
        for param, bounds in self.parameter_bounds.items():
            if random.random() < mutation_rate:
                # Mutation gaussienne
                sigma = (bounds[1] - bounds[0]) * 0.1
                new_value = solution.parameters[param] + random.gauss(0, sigma)
                solution.parameters[param] = max(bounds[0], min(bounds[1], new_value))

    def calculate_crowding_distance(self):
        """Calcule la distance de crowding pour chaque solution."""
        for front in self.fronts:
            if len(front) > 2:
                for objective in self.config.objectives:
                    front.sort(key=lambda x: x.objectives[objective.name])
                    
                    front[0].crowding_distance = float('inf')
                    front[-1].crowding_distance = float('inf')
                    
                    objective_range = (
                        front[-1].objectives[objective.name] -
                        front[0].objectives[objective.name]
                    )
                    
                    if objective_range == 0:
                        continue
                    
                    for i in range(1, len(front) - 1):
                        front[i].crowding_distance += (
                            front[i + 1].objectives[objective.name] -
                            front[i - 1].objectives[objective.name]
                        ) / objective_range

    def _record_history(self):
        """Enregistre l'historique de l'optimisation."""
        generation_stats = {
            'generation': self.generation,
            'pareto_front_size': len(self.fronts[0]),
            'objectives': {
                obj.name: {
                    'best': max(s.objectives[obj.name] for s in self.fronts[0]),
                    'avg': np.mean([s.objectives[obj.name] for s in self.population])
                }
                for obj in self.config.objectives
            }
        }
        self.history.append(generation_stats)

    def _log_progress(self):
        """Log les progrès de l'optimisation."""
        stats = self.history[-1]
        self.logger.info(
            f"Génération {stats['generation']} - "
            f"Taille du front de Pareto: {stats['pareto_front_size']}"
        )
        for obj_name, obj_stats in stats['objectives'].items():
            self.logger.info(
                f"{obj_name}: Best = {obj_stats['best']:.4f}, "
                f"Avg = {obj_stats['avg']:.4f}"
            )

    def get_optimization_summary(self) -> Dict:
        """Retourne un résumé de l'optimisation."""
        return {
            'total_generations': self.generation + 1,
            'final_pareto_front_size': len(self.fronts[0]),
            'objective_improvements': {
                obj.name: {
                    'initial': self.history[0]['objectives'][obj.name]['best'],
                    'final': self.history[-1]['objectives'][obj.name]['best'],
                    'improvement': (
                        self.history[-1]['objectives'][obj.name]['best'] -
                        self.history[0]['objectives'][obj.name]['best']
                    )
                }
                for obj in self.config.objectives
            },
            'execution_time': str(datetime.now() - self._start_time)
        } 