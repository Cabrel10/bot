"""
Module d'algorithme génétique pour l'optimisation des stratégies de trading.

Ce module implémente un algorithme génétique avancé pour optimiser les paramètres
des stratégies de trading. Il inclut des fonctionnalités comme :
- L'adaptation dynamique des taux de mutation et de croisement
- L'évaluation parallèle de la population
- La gestion de l'élitisme
- Le support multi-objectif
- La sauvegarde et le chargement des modèles
"""

import yaml
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from datetime import datetime
from dataclasses import dataclass, field
import random
import os
import json
from pathlib import Path

from trading.exceptions import (
    ValidationError,
    OptimizationError,
    ConfigurationError,
    DataError
)
from .fitness import FitnessEvaluator, FitnessMetrics
from .operators import (
    SelectionOperator,
    CrossoverOperator,
    MutationOperator
)

logger = logging.getLogger(__name__)

@dataclass
class GeneticParams:
    """
    Paramètres de configuration de l'algorithme génétique.
    
    Attributes:
        population_size (int): Taille de la population
        generations (int): Nombre maximum de générations
        mutation_rate (float): Taux initial de mutation [0,1]
        crossover_rate (float): Taux de croisement [0,1]
        elite_size (int): Nombre d'individus élites préservés
        tournament_size (int): Taille du tournoi pour la sélection
        convergence_threshold (float): Seuil de convergence
        max_stagnation (int): Nombre max de générations sans amélioration
        parallel_evaluation (bool): Active l'évaluation parallèle
        adaptive_rates (bool): Active l'adaptation des taux
        min_mutation_rate (float): Taux minimum de mutation
        max_mutation_rate (float): Taux maximum de mutation
        
    Raises:
        ValidationError: Si les paramètres sont invalides
    """
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnation: int = 15
    parallel_evaluation: bool = True
    adaptive_rates: bool = True
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.4
    
    def __post_init__(self):
        """Valide les paramètres après l'initialisation."""
        self._validate_params()
    
    def _validate_params(self):
        """Valide tous les paramètres de configuration."""
        if self.population_size < 10:
            raise ValidationError("La taille de population doit être >= 10")
            
        if self.generations < 1:
            raise ValidationError("Le nombre de générations doit être >= 1")
            
        if not 0 <= self.mutation_rate <= 1:
            raise ValidationError("Le taux de mutation doit être entre 0 et 1")
            
        if not 0 <= self.crossover_rate <= 1:
            raise ValidationError("Le taux de croisement doit être entre 0 et 1")
            
        if self.elite_size >= self.population_size:
            raise ValidationError("La taille d'élite doit être < population_size")
            
        if self.tournament_size >= self.population_size:
            raise ValidationError("La taille du tournoi doit être < population_size")
            
        if self.convergence_threshold <= 0:
            raise ValidationError("Le seuil de convergence doit être > 0")
            
        if not self.min_mutation_rate <= self.mutation_rate <= self.max_mutation_rate:
            raise ValidationError("Le taux de mutation doit être entre min et max")

@dataclass
class Individual:
    """
    Représente un individu dans la population.
    
    Attributes:
        genes (np.ndarray): Gènes de l'individu
        fitness (Optional[FitnessMetrics]): Métriques de fitness
        age (int): Âge en générations
        mutation_history (List[int]): Historique des mutations
        parent_fitness (List[float]): Fitness des parents
        metadata (Dict): Métadonnées additionnelles
    """
    genes: np.ndarray
    fitness: Optional[FitnessMetrics] = None
    age: int = 0
    mutation_history: List[int] = field(default_factory=list)
    parent_fitness: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> 'Individual':
        """
        Crée une copie profonde de l'individu.
        
        Returns:
            Individual: Nouvelle instance avec les mêmes attributs
        """
        new_ind = Individual(
            genes=self.genes.copy(),
            age=self.age,
            mutation_history=self.mutation_history.copy(),
            parent_fitness=self.parent_fitness.copy(),
            metadata=self.metadata.copy()
        )
        if self.fitness:
            new_ind.fitness = self.fitness
        return new_ind

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'individu en dictionnaire pour la sérialisation.
        
        Returns:
            Dict[str, Any]: Représentation de l'individu
        """
        return {
            'genes': self.genes.tolist(),
            'fitness': self.fitness.__dict__ if self.fitness else None,
            'age': self.age,
            'mutation_history': self.mutation_history,
            'parent_fitness': self.parent_fitness,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """
        Crée un individu à partir d'un dictionnaire.
        
        Args:
            data: Dictionnaire de données
            
        Returns:
            Individual: Nouvelle instance
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        try:
            genes = np.array(data['genes'])
            ind = cls(genes=genes)
            if data.get('fitness'):
                ind.fitness = FitnessMetrics(**data['fitness'])
            ind.age = data.get('age', 0)
            ind.mutation_history = data.get('mutation_history', [])
            ind.parent_fitness = data.get('parent_fitness', [])
            ind.metadata = data.get('metadata', {})
            return ind
        except (KeyError, ValueError) as e:
            raise ValidationError(f"Données d'individu invalides: {str(e)}")

class GeneticAlgorithm:
    """
    Algorithme génétique avancé pour l'optimisation des stratégies de trading.
    
    Cette classe implémente un algorithme génétique avec:
    - Adaptation dynamique des taux
    - Évaluation parallèle
    - Élitisme
    - Support multi-objectif
    - Sauvegarde/chargement des modèles
    
    Attributes:
        params (GeneticParams): Paramètres de configuration
        gene_bounds (Dict[str, Tuple[float, float]]): Bornes des gènes
        fitness_evaluator (FitnessEvaluator): Évaluateur de fitness
        population (List[Individual]): Population courante
        best_individual (Optional[Individual]): Meilleur individu
        generation (int): Génération courante
        history (List[Dict]): Historique d'évolution
    """

    def __init__(self, 
                 params: GeneticParams,
                 gene_bounds: Dict[str, Tuple[float, float]],
                 fitness_evaluator: FitnessEvaluator):
        """
        Initialise l'algorithme génétique.
        
        Args:
            params: Paramètres de configuration
            gene_bounds: Bornes pour chaque gène
            fitness_evaluator: Évaluateur de fitness
            
        Raises:
            ValidationError: Si les paramètres sont invalides
            ConfigurationError: Si la configuration est invalide
        """
        self._validate_initialization(params, gene_bounds, fitness_evaluator)
        
        self.params = params
        self.gene_bounds = gene_bounds
        self.fitness_evaluator = fitness_evaluator
        
        self.selection_op = SelectionOperator(tournament_size=params.tournament_size)
        self.crossover_op = CrossoverOperator()
        self.mutation_op = MutationOperator(
            mutation_rate=params.mutation_rate,
            min_rate=params.min_mutation_rate,
            max_rate=params.max_mutation_rate
        )
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.history: List[Dict] = []
        
        self._setup_logging()

    def _validate_initialization(self,
                               params: GeneticParams,
                               gene_bounds: Dict[str, Tuple[float, float]],
                               fitness_evaluator: FitnessEvaluator) -> None:
        """
        Valide les paramètres d'initialisation.
        
        Args:
            params: Paramètres de configuration
            gene_bounds: Bornes des gènes
            fitness_evaluator: Évaluateur de fitness
            
        Raises:
            ValidationError: Si les paramètres sont invalides
            ConfigurationError: Si la configuration est invalide
        """
        if not isinstance(params, GeneticParams):
            raise ValidationError("params doit être une instance de GeneticParams")
            
        if not gene_bounds:
            raise ValidationError("gene_bounds ne peut pas être vide")
            
        for key, (min_val, max_val) in gene_bounds.items():
            if min_val >= max_val:
                raise ValidationError(
                    f"Borne invalide pour {key}: min ({min_val}) >= max ({max_val})"
                )
                
        if not isinstance(fitness_evaluator, FitnessEvaluator):
            raise ValidationError(
                "fitness_evaluator doit être une instance de FitnessEvaluator"
            )

    def initialize_population(self) -> None:
        """
        Initialise la population avec des individus aléatoires.
        
        Raises:
            ValidationError: Si les bornes des gènes sont invalides
            OptimizationError: Si l'initialisation échoue
        """
        try:
            self.population = []
            gene_keys = list(self.gene_bounds.keys())
            
            for _ in range(self.params.population_size):
                try:
                    genes = np.array([
                        random.uniform(self.gene_bounds[key][0], self.gene_bounds[key][1])
                        for key in gene_keys
                    ])
                    self.population.append(Individual(genes))
                except Exception as e:
                    raise OptimizationError(
                        f"Erreur lors de la création d'un individu: {str(e)}"
                    )
                    
            logger.info(f"Population initialisée avec {len(self.population)} individus")
            
        except Exception as e:
            raise OptimizationError(f"Échec de l'initialisation: {str(e)}")

    def evolve(self, 
               market_data: np.ndarray,
               max_generations: Optional[int] = None) -> Individual:
        """
        Fait évoluer la population pour trouver la meilleure stratégie.
        
        Args:
            market_data: Données de marché pour l'évaluation
            max_generations: Nombre maximum de générations
        
        Returns:
            Individual: Meilleur individu trouvé
            
        Raises:
            ValidationError: Si les données sont invalides
            OptimizationError: Si l'évolution échoue
            DataError: Si les données de marché sont invalides
        """
        try:
            self._validate_market_data(market_data)
            
        if not self.population:
            self.initialize_population()

        max_gen = max_generations or self.params.generations
        stagnation_counter = 0
        best_fitness = float('-inf')
            last_improvement = 0

        for generation in range(max_gen):
                try:
            self.generation = generation
            
            # Évaluation de la population
            self._evaluate_population(market_data)
            
            # Mise à jour du meilleur individu
                    current_best = max(
                        self.population,
                        key=lambda x: x.fitness.sharpe_ratio if x.fitness else float('-inf')
                    )
                    
                    if (current_best.fitness and 
                        current_best.fitness.sharpe_ratio > best_fitness):
                best_fitness = current_best.fitness.sharpe_ratio
                self.best_individual = current_best.copy()
                        last_improvement = generation
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Vérification de la convergence
                    if (self._check_convergence() or 
                        stagnation_counter >= self.params.max_stagnation):
                        logger.info(
                            f"Convergence atteinte à la génération {generation}"
                            f" (stagnation: {stagnation_counter})"
                        )
                break

            # Création de la nouvelle génération
            self._create_next_generation()
            
            # Adaptation des taux si activée
            if self.params.adaptive_rates:
                self._adapt_rates()

            # Enregistrement de l'historique
            self._record_history()
                    
                    # Log de progression
                    if generation % 10 == 0:
                        self._log_progress(generation, max_gen)
                        
                except Exception as e:
                    logger.error(
                        f"Erreur à la génération {generation}: {str(e)}"
                    )
                    if generation - last_improvement > self.params.max_stagnation:
                        logger.warning(
                            "Arrêt anticipé dû à une stagnation prolongée"
                        )
                        break
                    continue

            if not self.best_individual:
                raise OptimizationError("Aucun individu valide trouvé")

        return self.best_individual
            
        except Exception as e:
            raise OptimizationError(f"Échec de l'évolution: {str(e)}")

    def _validate_market_data(self, market_data: np.ndarray) -> None:
        """
        Valide les données de marché.
        
        Args:
            market_data: Données à valider
            
        Raises:
            DataError: Si les données sont invalides
        """
        if not isinstance(market_data, np.ndarray):
            raise DataError("market_data doit être un numpy.ndarray")
            
        if len(market_data.shape) != 2:
            raise DataError("market_data doit être 2-dimensionnel")
            
        if np.any(np.isnan(market_data)) or np.any(np.isinf(market_data)):
            raise DataError("market_data contient des valeurs nan ou inf")
            
        if len(market_data) < self.params.population_size:
            raise DataError(
                "Pas assez de données pour la taille de population spécifiée"
            )

    def _evaluate_population(self, market_data: np.ndarray) -> None:
        """
        Évalue la fitness de toute la population.
        
        Args:
            market_data: Données de marché
            
        Raises:
            OptimizationError: Si l'évaluation échoue
        """
        try:
        if self.params.parallel_evaluation:
            with ThreadPoolExecutor() as executor:
                futures = [
                        executor.submit(
                            self._evaluate_individual_safe,
                            ind,
                            market_data
                        )
                    for ind in self.population
                ]
                for future in futures:
                    future.result()
        else:
            for ind in self.population:
                    self._evaluate_individual_safe(ind, market_data)

        except Exception as e:
            raise OptimizationError(f"Échec de l'évaluation: {str(e)}")

    def _evaluate_individual_safe(self, 
                           individual: Individual, 
                           market_data: np.ndarray) -> None:
        """
        Évalue un individu avec gestion d'erreurs.
        
        Args:
            individual: Individu à évaluer
            market_data: Données de marché
            
        Raises:
            OptimizationError: Si l'évaluation échoue
        """
        try:
        predictions = self._get_predictions(individual.genes, market_data)
        individual.fitness = self.fitness_evaluator.calculate_fitness(
            predictions=predictions,
            actual_returns=market_data['returns']
        )
        individual.age += 1
            
        except Exception as e:
            logger.warning(f"Échec de l'évaluation d'un individu: {str(e)}")
            individual.fitness = None

    def _create_next_generation(self) -> None:
        """
        Crée la prochaine génération de la population.
        
        Raises:
            OptimizationError: Si la création échoue
        """
        try:
        new_population = []
        
        # Élitisme
        sorted_pop = sorted(
                [ind for ind in self.population if ind.fitness is not None],
            key=lambda x: x.fitness.sharpe_ratio,
            reverse=True
        )
            
            if not sorted_pop:
                raise OptimizationError("Aucun individu valide dans la population")
                
            new_population.extend(
                ind.copy() for ind in sorted_pop[:self.params.elite_size]
            )
        
        # Création du reste de la population
        while len(new_population) < self.params.population_size:
                try:
            if random.random() < self.params.crossover_rate:
                        parent1 = self.selection_op.select(sorted_pop)
                        parent2 = self.selection_op.select(sorted_pop)
                child1, child2 = self.crossover_op.crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                        parent = self.selection_op.select(sorted_pop)
                child = parent.copy()
                new_population.append(child)
                except Exception as e:
                    logger.warning(f"Échec de création d'enfant: {str(e)}")
                    continue
        
        # Application des mutations
        for ind in new_population[self.params.elite_size:]:
                try:
            self.mutation_op.mutate(ind, self.gene_bounds)
                except Exception as e:
                    logger.warning(f"Échec de mutation: {str(e)}")
        
        self.population = new_population[:self.params.population_size]
            
        except Exception as e:
            raise OptimizationError(f"Échec de création de génération: {str(e)}")

    def _adapt_rates(self) -> None:
        """
        Adapte les taux de mutation et de croisement.
        
        Cette méthode ajuste dynamiquement les taux en fonction de la diversité
        de la population et de la convergence.
        """
        try:
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                return
                
            fitnesses = [ind.fitness.sharpe_ratio for ind in valid_individuals]
            avg_fitness = np.mean(fitnesses)
            best_fitness = max(fitnesses)
            
            # Calcul de la diversité
            diversity = np.std([
                np.std(ind.genes) for ind in valid_individuals
            ])
            
            # Ajustement des taux
        if best_fitness - avg_fitness < self.params.convergence_threshold:
            # Augmente la mutation si la population converge
                self.mutation_op.mutation_rate = min(
                    self.params.max_mutation_rate,
                    self.mutation_op.mutation_rate * 1.5
                )
                self.params.crossover_rate = max(
                    0.5,
                    self.params.crossover_rate * 0.9
                )
        else:
            # Réduit la mutation si la population est diverse
                self.mutation_op.mutation_rate = max(
                    self.params.min_mutation_rate,
                    self.mutation_op.mutation_rate * 0.9
                )
                self.params.crossover_rate = min(
                    0.95,
                    self.params.crossover_rate * 1.1
                )
                
            logger.debug(
                f"Taux adaptés - mutation: {self.mutation_op.mutation_rate:.3f}, "
                f"crossover: {self.params.crossover_rate:.3f}"
            )
            
        except Exception as e:
            logger.warning(f"Échec de l'adaptation des taux: {str(e)}")

    def _check_convergence(self) -> bool:
        """
        Vérifie si l'algorithme a convergé.
        
        Returns:
            bool: True si convergé
        """
        try:
        if len(self.history) < 2:
            return False
            
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                return False
                
            current_best = max(
                ind.fitness.sharpe_ratio for ind in valid_individuals
            )
        prev_best = self.history[-1]['best_fitness']
        
            # Vérifie aussi la diversité génétique
            gene_diversity = np.std([
                np.std(ind.genes) for ind in valid_individuals
            ])
            
            return (
                abs(current_best - prev_best) < self.params.convergence_threshold
                and gene_diversity < self.params.convergence_threshold
            )
            
        except Exception as e:
            logger.warning(f"Erreur lors du check de convergence: {str(e)}")
            return False

    def _record_history(self) -> None:
        """
        Enregistre l'historique de l'évolution.
        
        Cette méthode capture les métriques importantes de chaque génération
        pour le suivi et la visualisation.
        """
        try:
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                return
                
            fitnesses = [ind.fitness.sharpe_ratio for ind in valid_individuals]
            
        generation_stats = {
            'generation': self.generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'mutation_rate': self.mutation_op.mutation_rate,
                'crossover_rate': self.params.crossover_rate,
                'valid_individuals': len(valid_individuals),
                'gene_diversity': np.std([
                    np.std(ind.genes) for ind in valid_individuals
                ])
            }
            
        self.history.append(generation_stats)

        except Exception as e:
            logger.warning(f"Erreur lors de l'enregistrement historique: {str(e)}")

    def _log_progress(self, generation: int, max_gen: int) -> None:
        """
        Log la progression de l'évolution.
        
        Args:
            generation: Génération courante
            max_gen: Nombre maximum de générations
        """
        try:
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                logger.warning(
                    f"Génération {generation}/{max_gen}: "
                    "Aucun individu valide"
                )
                return
                
            best_fitness = max(
                ind.fitness.sharpe_ratio for ind in valid_individuals
            )
            avg_fitness = np.mean([
                ind.fitness.sharpe_ratio for ind in valid_individuals
            ])
            
            logger.info(
                f"Génération {generation}/{max_gen} - "
                f"Meilleur: {best_fitness:.4f}, "
                f"Moyenne: {avg_fitness:.4f}, "
                f"Valides: {len(valid_individuals)}/{len(self.population)}"
            )
            
        except Exception as e:
            logger.warning(f"Erreur lors du log de progression: {str(e)}")

    def _setup_logging(self) -> None:
        """Configure le système de logging."""
        try:
            log_format = (
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        logging.basicConfig(
                level=logging.INFO,
                format=log_format
            )
            
            # Ajout d'un handler pour fichier si nécessaire
            if hasattr(self.params, 'log_file'):
                file_handler = logging.FileHandler(self.params.log_file)
                file_handler.setFormatter(logging.Formatter(log_format))
                logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"Erreur lors de la configuration du logging: {str(e)}")

    def _get_predictions(self, genes: np.ndarray, market_data: np.ndarray) -> np.ndarray:
        """
        Obtient les prédictions à partir des gènes et des données de marché.
        À implémenter selon la stratégie spécifique.
        """
        raise NotImplementedError("Cette méthode doit être implémentée dans une sous-classe")
    
    def calculate_fitness(self, individual: Dict, data: np.ndarray) -> float:
        """
        Calcule le score de fitness pour un individu.
        
        Args:
            individual: Dictionnaire des paramètres de l'individu
            data: Données de marché pour l'évaluation
            
        Returns:
            float: Score de fitness
            
        Raises:
            ValidationError: Si les données sont invalides
            OptimizationError: Si le calcul échoue
        """
        try:
            if not isinstance(individual, dict):
                raise ValidationError("individual doit être un dictionnaire")
                
            if not isinstance(data, np.ndarray):
                raise ValidationError("data doit être un numpy.ndarray")
                
            # Validation des contraintes
        if not self._check_constraints(individual, data):
                return float('-inf')
                
            # Calcul des métriques
            metrics = self.fitness_evaluator.calculate_fitness(
                predictions=self._get_predictions(
                    np.array(list(individual.values())),
                    data
                ),
                actual_returns=data['returns']
            )
            
            # Calcul du score pondéré
            score = sum(
                self.params.weights.get(metric, 0) * value
                for metric, value in metrics.__dict__.items()
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du fitness: {str(e)}")
            return float('-inf')
    
    def _check_constraints(self, individual: Dict, data: np.ndarray) -> bool:
        """
        Vérifie si un individu respecte les contraintes.
        
        Args:
            individual: Dictionnaire des paramètres
            data: Données de marché
            
        Returns:
            bool: True si les contraintes sont respectées
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        try:
            # Validation des types
            if not isinstance(individual, dict):
                raise ValidationError("individual doit être un dictionnaire")
                
            if not isinstance(data, np.ndarray):
                raise ValidationError("data doit être un numpy.ndarray")
                
            # Vérification des bornes
            for key, value in individual.items():
                if key not in self.gene_bounds:
                    raise ValidationError(f"Gène inconnu: {key}")
                    
                min_val, max_val = self.gene_bounds[key]
                if not min_val <= value <= max_val:
                    logger.debug(
                        f"Contrainte de borne violée pour {key}: "
                        f"{value} not in [{min_val}, {max_val}]"
                    )
                    return False
                    
            # Vérification des contraintes spécifiques
            predictions = self._get_predictions(
                np.array(list(individual.values())),
                data
            )
            metrics = self.fitness_evaluator.calculate_fitness(
                predictions=predictions,
                actual_returns=data['returns']
            )
            
            # Vérification du drawdown maximum
            if (metrics.max_drawdown < 
                self.params.constraints.get('max_drawdown', float('-inf'))):
                logger.debug(
                    f"Contrainte de drawdown violée: "
                    f"{metrics.max_drawdown:.2%}"
                )
                return False
        
        # Vérification du nombre minimum de trades
            if (metrics.total_trades < 
                self.params.constraints.get('min_trades', 0)):
                logger.debug(
                    f"Contrainte de trades violée: "
                    f"{metrics.total_trades} trades"
                )
            return False
        
            # Vérification du ratio de Sharpe minimum
            if (metrics.sharpe_ratio < 
                self.params.constraints.get('min_sharpe', float('-inf'))):
                logger.debug(
                    f"Contrainte de Sharpe violée: "
                    f"{metrics.sharpe_ratio:.2f}"
                )
            return False
        
        return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des contraintes: {str(e)}")
            return False
    
    def select_parents(self) -> List[Dict]:
        """
        Sélectionne les parents pour la reproduction.
        
        Returns:
            List[Dict]: Liste des parents sélectionnés
            
        Raises:
            OptimizationError: Si la sélection échoue
        """
        try:
            if not self.population:
                raise OptimizationError("Population vide")
                
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                raise OptimizationError("Aucun individu valide")
                
            # Sélection par tournoi
            parents = self._tournament_selection()
            
            if len(parents) < 2:
                raise OptimizationError(
                    f"Pas assez de parents sélectionnés: {len(parents)}"
                )
                
            return parents
            
        except Exception as e:
            raise OptimizationError(f"Échec de la sélection: {str(e)}")
    
    def _tournament_selection(self) -> List[Dict]:
        """
        Implémente la sélection par tournoi.
        
        Returns:
            List[Dict]: Parents sélectionnés
            
        Raises:
            OptimizationError: Si la sélection échoue
        """
        try:
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                raise OptimizationError("Aucun individu valide")
                
            def select_one() -> Dict:
                """Sélectionne un individu par tournoi."""
                tournament = random.sample(
                    valid_individuals,
                    min(self.params.tournament_size, len(valid_individuals))
                )
                return max(
                    tournament,
                    key=lambda x: x.fitness.sharpe_ratio
                ).to_dict()
                
            return [select_one() for _ in range(2)]
            
        except Exception as e:
            raise OptimizationError(f"Échec du tournoi: {str(e)}")
    
    def crossover(self, parents: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Effectue le croisement entre deux parents.
        
        Args:
            parents: Liste des parents
            
        Returns:
            Tuple[Dict, Dict]: Enfants générés
            
        Raises:
            ValidationError: Si les parents sont invalides
            OptimizationError: Si le croisement échoue
        """
        try:
            if not isinstance(parents, list) or len(parents) != 2:
                raise ValidationError("Besoin de exactement 2 parents")
        
        parent1, parent2 = parents
            if not all(isinstance(p, dict) for p in parents):
                raise ValidationError("Les parents doivent être des dictionnaires")
                
            # Validation des gènes
            for parent in parents:
                missing = set(self.gene_bounds.keys()) - set(parent.keys())
                if missing:
                    raise ValidationError(f"Gènes manquants: {missing}")
                    
            # Croisement uniforme
            child1, child2 = {}, {}
            for gene in self.gene_bounds:
                if random.random() < 0.5:
                    child1[gene] = parent1[gene]
                    child2[gene] = parent2[gene]
            else:
                    child1[gene] = parent2[gene]
                    child2[gene] = parent1[gene]
        
        return child1, child2
            
        except Exception as e:
            raise OptimizationError(f"Échec du croisement: {str(e)}")
    
    def mutate(self, individual: Dict) -> Dict:
        """
        Applique une mutation à un individu.
        
        Args:
            individual: Individu à muter
            
        Returns:
            Dict: Individu muté
            
        Raises:
            ValidationError: Si l'individu est invalide
            OptimizationError: Si la mutation échoue
        """
        try:
            if not isinstance(individual, dict):
                raise ValidationError("individual doit être un dictionnaire")
                
            # Validation des gènes
            missing = set(self.gene_bounds.keys()) - set(individual.keys())
            if missing:
                raise ValidationError(f"Gènes manquants: {missing}")
                
            # Copie de l'individu
        mutated = individual.copy()
            
            # Application des mutations
            for gene in self.gene_bounds:
                if random.random() < self.mutation_op.mutation_rate:
                    min_val, max_val = self.gene_bounds[gene]
                    # Mutation gaussienne
                    sigma = (max_val - min_val) * 0.1
                    value = mutated[gene] + random.gauss(0, sigma)
                    # Clip aux bornes
                    mutated[gene] = max(min_val, min(max_val, value))
        
        return mutated
            
        except Exception as e:
            raise OptimizationError(f"Échec de la mutation: {str(e)}")
    
    def _plot_evolution(self) -> None:
        """
        Génère des graphiques de l'évolution de l'algorithme.
        
        Cette méthode crée plusieurs visualisations :
        - Évolution du meilleur fitness
        - Évolution de la moyenne et écart-type
        - Évolution des taux de mutation/croisement
        - Diversité génétique
        
        Raises:
            IOError: Si la sauvegarde des graphiques échoue
        """
        try:
            if not self.history:
                logger.warning("Pas d'historique à visualiser")
                return
                
            # Configuration des sous-plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Évolution de l'algorithme génétique")
            
            # Évolution du fitness
            ax = axes[0, 0]
            generations = [gen['generation'] for gen in self.history]
            best_fitness = [gen['best_fitness'] for gen in self.history]
            avg_fitness = [gen['avg_fitness'] for gen in self.history]
            std_fitness = [gen['std_fitness'] for gen in self.history]
            
            ax.plot(generations, best_fitness, 'b-', label='Meilleur')
            ax.plot(generations, avg_fitness, 'g-', label='Moyenne')
            ax.fill_between(
                generations,
                [avg - std for avg, std in zip(avg_fitness, std_fitness)],
                [avg + std for avg, std in zip(avg_fitness, std_fitness)],
                alpha=0.2
            )
            ax.set_xlabel('Génération')
            ax.set_ylabel('Fitness')
            ax.legend()
            
            # Évolution des taux
            ax = axes[0, 1]
            mutation_rates = [gen['mutation_rate'] for gen in self.history]
            crossover_rates = [
                gen.get('crossover_rate', self.params.crossover_rate)
                for gen in self.history
            ]
            
            ax.plot(generations, mutation_rates, 'r-', label='Mutation')
            ax.plot(generations, crossover_rates, 'y-', label='Croisement')
            ax.set_xlabel('Génération')
            ax.set_ylabel('Taux')
            ax.legend()
            
            # Diversité génétique
            ax = axes[1, 0]
            diversity = [gen.get('gene_diversity', 0) for gen in self.history]
            ax.plot(generations, diversity, 'g-')
            ax.set_xlabel('Génération')
            ax.set_ylabel('Diversité génétique')
            
            # Individus valides
            ax = axes[1, 1]
            valid_counts = [
                gen.get('valid_individuals', len(self.population))
                for gen in self.history
            ]
            ax.plot(generations, valid_counts, 'm-')
            ax.set_xlabel('Génération')
            ax.set_ylabel('Individus valides')
            
            # Sauvegarde
            plt.tight_layout()
            plot_path = 'evolution.png'
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Graphiques sauvegardés dans {plot_path}")
            
        except Exception as e:
            raise IOError(f"Erreur lors de la création des graphiques: {str(e)}")
    
    def save_best_individual(self, path: str) -> None:
        """
        Sauvegarde le meilleur individu dans un fichier.
        
        Args:
            path: Chemin du fichier de sauvegarde
            
        Raises:
            ValidationError: Si le meilleur individu n'existe pas
            IOError: Si la sauvegarde échoue
        """
        try:
            if not self.best_individual:
                raise ValidationError("Pas de meilleur individu à sauvegarder")
                
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'genes': self.best_individual.genes.tolist(),
                'fitness': self.best_individual.fitness.__dict__,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'generation': self.generation,
                    'params': self.params.__dict__
                }
            }
        
        with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Meilleur individu sauvegardé dans {path}")
            
        except Exception as e:
            raise IOError(f"Erreur lors de la sauvegarde: {str(e)}")
    
    @staticmethod
    def load_individual(path: str) -> Dict:
        """
        Charge un individu depuis un fichier.
        
        Args:
            path: Chemin du fichier
            
        Returns:
            Dict: Données de l'individu
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValidationError: Si les données sont invalides
            IOError: Si le chargement échoue
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {path}")
                
            with open(path) as f:
                data = json.load(f)
                
            # Validation des données
            required_fields = {'genes', 'fitness', 'metadata'}
            missing = required_fields - set(data.keys())
            if missing:
                raise ValidationError(f"Champs manquants: {missing}")
                
            # Conversion en numpy array
            data['genes'] = np.array(data['genes'])
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Format de fichier invalide: {str(e)}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement: {str(e)}")

    def get_population_stats(self) -> Dict[str, Any]:
        """
        Calcule des statistiques sur la population courante.
        
        Returns:
            Dict[str, Any]: Statistiques calculées
            
        Les statistiques incluent:
        - Taille de la population
        - Nombre d'individus valides
        - Meilleur fitness
        - Fitness moyen
        - Écart-type du fitness
        - Diversité génétique
        - Âge moyen
        - Distribution des mutations
        """
        try:
            valid_individuals = [
                ind for ind in self.population 
                if ind.fitness is not None
            ]
            
            if not valid_individuals:
                return {
                    'population_size': len(self.population),
                    'valid_count': 0,
                    'error': 'Aucun individu valide'
                }
                
            fitnesses = [ind.fitness.sharpe_ratio for ind in valid_individuals]
            genes = np.array([ind.genes for ind in valid_individuals])
            ages = [ind.age for ind in valid_individuals]
            mutation_counts = [len(ind.mutation_history) for ind in valid_individuals]
            
            return {
                'population_size': len(self.population),
                'valid_count': len(valid_individuals),
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'gene_diversity': np.std(genes, axis=0).mean(),
                'avg_age': np.mean(ages),
                'max_age': max(ages),
                'avg_mutations': np.mean(mutation_counts),
                'mutation_distribution': np.histogram(
                    mutation_counts,
                    bins='auto'
                )[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {str(e)}")
            return {'error': str(e)}

    def save_state(self, path: Union[str, Path]) -> None:
        """
        Sauvegarde l'état complet de l'algorithme.
        
        Args:
            path: Chemin de sauvegarde
            
        Raises:
            IOError: Si la sauvegarde échoue
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'params': self.params.__dict__,
                'gene_bounds': self.gene_bounds,
                'generation': self.generation,
                'population': [ind.to_dict() for ind in self.population],
                'best_individual': (
                    self.best_individual.to_dict() 
                    if self.best_individual else None
                ),
                'history': self.history
            }
            
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"État sauvegardé dans {path}")
            
        except Exception as e:
            raise IOError(f"Erreur lors de la sauvegarde: {str(e)}")

    @classmethod
    def load_state(cls, path: Union[str, Path]) -> 'GeneticAlgorithm':
        """
        Charge l'état d'un algorithme précédemment sauvegardé.
        
        Args:
            path: Chemin du fichier d'état
            
        Returns:
            GeneticAlgorithm: Instance restaurée
            
        Raises:
            IOError: Si le chargement échoue
            ValidationError: Si les données sont invalides
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {path}")
                
            with open(path) as f:
                state = json.load(f)
                
            params = GeneticParams(**state['params'])
            gene_bounds = state['gene_bounds']
            
            # Création d'une nouvelle instance
            algo = cls(
                params=params,
                gene_bounds=gene_bounds,
                fitness_evaluator=FitnessEvaluator(params)
            )
            
            # Restauration de l'état
            algo.generation = state['generation']
            algo.population = [
                Individual.from_dict(ind_data)
                for ind_data in state['population']
            ]
            if state['best_individual']:
                algo.best_individual = Individual.from_dict(
                    state['best_individual']
                )
            algo.history = state['history']
            
            logger.info(f"État chargé depuis {path}")
            return algo
            
        except Exception as e:
            raise IOError(f"Erreur lors du chargement: {str(e)}")

    def reset(self) -> None:
        """
        Réinitialise l'algorithme à son état initial.
        
        Cette méthode:
        - Vide la population
        - Réinitialise les compteurs
        - Efface l'historique
        - Réinitialise les taux
        """
        try:
            self.population = []
            self.generation = 0
            self.best_individual = None
            self.history = []
            
            # Réinitialisation des taux
            self.mutation_op.mutation_rate = self.params.mutation_rate
            self.params.crossover_rate = self.params.crossover_rate
            
            logger.info("Algorithme réinitialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation: {str(e)}")
            raise OptimizationError(f"Échec de la réinitialisation: {str(e)}")

    def validate(self) -> bool:
        """
        Valide l'état de l'algorithme.
        
        Returns:
            bool: True si l'état est valide
            
        Cette méthode vérifie:
        - La cohérence de la population
        - La validité des paramètres
        - L'intégrité des données
        """
        try:
            # Validation de la population
            if self.population:
                if len(self.population) != self.params.population_size:
                    logger.error("Taille de population incorrecte")
                    return False
                    
                for ind in self.population:
                    if not isinstance(ind, Individual):
                        logger.error("Type d'individu invalide")
                        return False
                        
                    if len(ind.genes) != len(self.gene_bounds):
                        logger.error("Nombre de gènes incorrect")
                        return False
                        
            # Validation des paramètres
            if not self.params.validate():
                logger.error("Paramètres invalides")
                return False
                
            # Validation de l'historique
            if self.history:
                required_fields = {
                    'generation', 'best_fitness', 'avg_fitness',
                    'std_fitness', 'mutation_rate'
                }
                for entry in self.history:
                    missing = required_fields - set(entry.keys())
                    if missing:
                        logger.error(f"Champs manquants dans l'historique: {missing}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return False

    def __str__(self) -> str:
        """
        Retourne une représentation textuelle de l'algorithme.
        
        Returns:
            str: Description de l'état
        """
        try:
            valid_count = len([
                ind for ind in self.population 
                if ind.fitness is not None
            ])
            
            best_fitness = (
                self.best_individual.fitness.sharpe_ratio
                if self.best_individual and self.best_individual.fitness
                else None
            )
            
            return (
                f"AlgorithmeGénétique(génération={self.generation}, "
                f"population={len(self.population)}, "
                f"valides={valid_count}, "
                f"meilleur_fitness={best_fitness:.4f if best_fitness else None})"
            )
            
        except Exception as e:
            return f"AlgorithmeGénétique(erreur={str(e)})"

    def __repr__(self) -> str:
        """
        Retourne une représentation détaillée de l'algorithme.
        
        Returns:
            str: Représentation détaillée
        """
        try:
            return (
                f"GeneticAlgorithm(\n"
                f"  params={self.params},\n"
                f"  gene_bounds={self.gene_bounds},\n"
                f"  generation={self.generation},\n"
                f"  population_size={len(self.population)},\n"
                f"  best_fitness={self.best_individual.fitness if self.best_individual else None}\n"
                f")"
            )
            
        except Exception as e:
            return f"GeneticAlgorithm(error={str(e)})"