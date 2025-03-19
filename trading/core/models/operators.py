from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import random
from .genetic_algorithm import Individual

@dataclass
class GeneticOperatorConfig:
    """Configuration des opérateurs génétiques"""
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    mutation_strength: float = 0.2
    tournament_size: int = 3
    elitism_count: int = 2

class GeneticOperators:
    """Implémentation des opérateurs génétiques pour l'optimisation"""
    
    def __init__(self, config: Union[Dict, GeneticOperatorConfig]):
        """
        Initialise les opérateurs génétiques.
        
        Args:
            config: Configuration des opérateurs contenant:
                - mutation_rate: Taux de mutation
                - crossover_rate: Taux de croisement
                - mutation_strength: Force de la mutation
                - tournament_size: Taille du tournoi pour la sélection
                - elitism_count: Nombre d'élites à préserver
        """
        if isinstance(config, dict):
            self.config = GeneticOperatorConfig(
                mutation_rate=config.get('mutation_rate', 0.1),
                crossover_rate=config.get('crossover_rate', 0.8),
                mutation_strength=config.get('mutation_strength', 0.2),
                tournament_size=config.get('tournament_size', 3),
                elitism_count=config.get('elitism_count', 2)
            )
        else:
            self.config = config
    
    def tournament_selection(self, population: List[Dict], 
                           fitness_scores: np.ndarray) -> List[Dict]:
        """
        Sélectionne les parents par tournoi.
        
        Args:
            population: Liste des individus
            fitness_scores: Scores de fitness correspondants
            
        Returns:
            List[Dict]: Parents sélectionnés
        """
        selected = []
        pop_size = len(population)
        
        while len(selected) < pop_size - self.config.elitism_count:
            # Sélection des participants au tournoi
            tournament_idx = np.random.choice(
                pop_size,
                size=self.config.tournament_size,
                replace=False
            )
            tournament_fitness = fitness_scores[tournament_idx]
            
            # Sélection du vainqueur
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Ajout des élites
        elite_idx = np.argsort(fitness_scores)[-self.config.elitism_count:]
        selected.extend([population[i].copy() for i in elite_idx])
        
        return selected
    
    def uniform_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Effectue un croisement uniforme entre deux parents.
        
        Args:
            parent1: Premier parent
            parent2: Second parent
            
        Returns:
            Tuple[Dict, Dict]: Deux enfants générés
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2
    
    def blend_crossover(self, parent1: Dict, parent2: Dict, alpha: float = 0.5) -> Tuple[Dict, Dict]:
        """
        Effectue un croisement par mélange (BLX-alpha).
        
        Args:
            parent1: Premier parent
            parent2: Second parent
            alpha: Paramètre de mélange
            
        Returns:
            Tuple[Dict, Dict]: Deux enfants générés
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        
        for key in parent1.keys():
            # Calcul des bornes pour le mélange
            min_val = min(parent1[key], parent2[key])
            max_val = max(parent1[key], parent2[key])
            range_val = max_val - min_val
            
            # Extension des bornes
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            # Génération des valeurs des enfants
            child1[key] = random.uniform(lower, upper)
            child2[key] = random.uniform(lower, upper)
        
        return child1, child2
    
    def gaussian_mutation(self, individual: Dict) -> Dict:
        """
        Applique une mutation gaussienne aux paramètres.
        
        Args:
            individual: Individu à muter
            
        Returns:
            Dict: Individu muté
        """
        mutated = individual.copy()
        
        for key in mutated.keys():
            if random.random() < self.config.mutation_rate:
                # Mutation gaussienne
                noise = np.random.normal(0, self.config.mutation_strength)
                mutated[key] *= (1 + noise)
        
        return mutated
    
    def adaptive_mutation(self, individual: Dict, generation: int,
                         max_generations: int) -> Dict:
        """
        Applique une mutation avec force adaptative.
        
        Args:
            individual: Individu à muter
            generation: Génération actuelle
            max_generations: Nombre maximum de générations
            
        Returns:
            Dict: Individu muté
        """
        mutated = individual.copy()
        
        # Adaptation de la force de mutation
        progress = generation / max_generations
        adaptive_strength = self.config.mutation_strength * (1 - progress)
        
        for key in mutated.keys():
            if random.random() < self.config.mutation_rate:
                noise = np.random.normal(0, adaptive_strength)
                mutated[key] *= (1 + noise)
        
        return mutated
    
    def differential_mutation(self, population: List[Dict],
                            target_idx: int,
                            scale: float = 0.5) -> Dict:
        """
        Applique une mutation différentielle.
        
        Args:
            population: Population actuelle
            target_idx: Index de l'individu cible
            scale: Facteur d'échelle pour la mutation
            
        Returns:
            Dict: Individu muté
        """
        # Sélection aléatoire de trois individus différents
        available_idx = list(range(len(population)))
        available_idx.remove(target_idx)
        a, b, c = random.sample(available_idx, 3)
        
        base = population[a].copy()
        mutated = {}
        
        for key in base.keys():
            # Mutation différentielle: base + scale * (b - c)
            mutated[key] = base[key] + scale * (
                population[b][key] - population[c][key]
            )
        
        return mutated
    
    def create_next_generation(self, population: List[Dict],
                             fitness_scores: np.ndarray,
                             generation: int,
                             max_generations: int) -> List[Dict]:
        """
        Crée la prochaine génération d'individus.
        
        Args:
            population: Population actuelle
            fitness_scores: Scores de fitness
            generation: Génération actuelle
            max_generations: Nombre maximum de générations
            
        Returns:
            List[Dict]: Nouvelle génération
        """
        # Sélection des parents
        parents = self.tournament_selection(population, fitness_scores)
        
        # Création de la nouvelle génération
        new_generation = []
        
        # Préservation des élites
        elite_idx = np.argsort(fitness_scores)[-self.config.elitism_count:]
        new_generation.extend([population[i].copy() for i in elite_idx])
        
        # Croisement et mutation pour le reste de la population
        while len(new_generation) < len(population):
            # Sélection des parents pour le croisement
            parent1, parent2 = random.sample(parents, 2)
            
            # Croisement
            child1, child2 = self.blend_crossover(parent1, parent2)
            
            # Mutation
            child1 = self.adaptive_mutation(child1, generation, max_generations)
            child2 = self.adaptive_mutation(child2, generation, max_generations)
            
            new_generation.extend([child1, child2])
        
        # Ajustement de la taille si nécessaire
        if len(new_generation) > len(population):
            new_generation = new_generation[:len(population)]
        
        return new_generation

class SelectionOperator:
    """Opérateur de sélection pour l'algorithme génétique."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual]) -> Individual:
        """
        Sélectionne un individu par tournoi.
        
        Args:
            population: Liste des individus
        
        Returns:
            Individual: Individu sélectionné
        """
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness.sharpe_ratio)

class CrossoverOperator:
    """Opérateur de croisement pour l'algorithme génétique."""
    
    def crossover(self, 
                 parent1: Individual, 
                 parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Effectue un croisement entre deux parents.
        
        Args:
            parent1: Premier parent
            parent2: Deuxième parent
        
        Returns:
            Tuple[Individual, Individual]: Deux enfants
        """
        # Croisement arithmétique
        alpha = random.random()
        genes1 = alpha * parent1.genes + (1 - alpha) * parent2.genes
        genes2 = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        child1 = Individual(genes1)
        child2 = Individual(genes2)
        
        # Transmission de l'historique
        child1.parent_fitness = [parent1.fitness.sharpe_ratio]
        child2.parent_fitness = [parent2.fitness.sharpe_ratio]
        
        return child1, child2

    def adaptive_crossover(self, 
                         parent1: Individual, 
                         parent2: Individual,
                         population_stats: Dict) -> Tuple[Individual, Individual]:
        """
        Croisement adaptatif basé sur les statistiques de la population.
        
        Args:
            parent1: Premier parent
            parent2: Deuxième parent
            population_stats: Statistiques de la population
        
        Returns:
            Tuple[Individual, Individual]: Deux enfants
        """
        # Calcul du taux de croisement adaptatif
        fitness_range = population_stats['max_fitness'] - population_stats['min_fitness']
        if fitness_range > 0:
            alpha = (parent1.fitness.sharpe_ratio - population_stats['min_fitness']) / fitness_range
        else:
            alpha = 0.5
            
        # Croisement avec le taux adaptatif
        genes1 = alpha * parent1.genes + (1 - alpha) * parent2.genes
        genes2 = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        child1 = Individual(genes1)
        child2 = Individual(genes2)
        
        return child1, child2

class MutationOperator:
    """Opérateur de mutation pour l'algorithme génétique."""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
        self.mutation_strength = 0.1  # Force initiale de la mutation

    def mutate(self, 
               individual: Individual, 
               gene_bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Applique une mutation à un individu.
        
        Args:
            individual: Individu à muter
            gene_bounds: Bornes pour chaque gène
        """
        for i in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                # Mutation gaussienne adaptative
                sigma = self.mutation_strength * (gene_bounds[list(gene_bounds.keys())[i]][1] - 
                                               gene_bounds[list(gene_bounds.keys())[i]][0])
                mutation = np.random.normal(0, sigma)
                
                # Application de la mutation avec respect des bornes
                new_value = individual.genes[i] + mutation
                lower_bound = gene_bounds[list(gene_bounds.keys())[i]][0]
                upper_bound = gene_bounds[list(gene_bounds.keys())[i]][1]
                individual.genes[i] = np.clip(new_value, lower_bound, upper_bound)
                
                # Enregistrement de la mutation
                individual.mutation_history.append(i)

    def adaptive_mutation(self, 
                        individual: Individual,
                        gene_bounds: Dict[str, Tuple[float, float]],
                        population_stats: Dict) -> None:
        """
        Mutation adaptative basée sur les statistiques de la population.
        
        Args:
            individual: Individu à muter
            gene_bounds: Bornes pour chaque gène
            population_stats: Statistiques de la population
        """
        # Ajustement du taux de mutation basé sur la diversité de la population
        diversity = population_stats['std_fitness'] / population_stats['avg_fitness']
        self.mutation_rate = max(0.01, min(0.4, 0.1 / diversity))
        
        # Ajustement de la force de mutation basé sur l'âge de l'individu
        self.mutation_strength = 0.1 * (1 + individual.age / 10)
        
        self.mutate(individual, gene_bounds) 