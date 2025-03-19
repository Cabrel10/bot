from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid

from .chromosome import TradingChromosome
from ...data.data_types import ProcessedData, TradeData
from ...utils.logger import TradingLogger

@dataclass
class TradeHistory:
    """Historique des trades d'un individu."""
    trades: List[TradeData]
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float

@dataclass
class IndividualStats:
    """Statistiques de performance d'un individu."""
    fitness_score: float = 0.0
    trade_history: Optional[TradeHistory] = None
    generation_created: int = 0
    lifetime_trades: int = 0
    best_performance: float = float('-inf')
    worst_performance: float = float('inf')
    avg_performance: float = 0.0
    performance_history: List[float] = field(default_factory=list)

class Individual:
    """Représente un individu dans un algorithme génétique."""
    
    def __init__(self, genes: list, fitness: float = 0.0):
        """
        Initialise un individu.
        
        Args:
            genes: Liste des gènes de l'individu.
            fitness: Valeur de fitness de l'individu.
        """
        self.genes = genes
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness})"

class TradingIndividual:
    """Représente un individu dans la population de trading."""

    def __init__(self, chromosome: Optional[TradingChromosome] = None):
        """Initialise un individu.
        
        Args:
            chromosome: Chromosome optionnel (sinon créé aléatoirement)
        """
        self.id = str(uuid.uuid4())
        self.chromosome = chromosome or TradingChromosome()
        self.stats = IndividualStats()
        self.logger = TradingLogger()
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.now().isoformat(),
            'parent_ids': [],
            'mutation_history': [],
            'crossover_history': []
        }

    async def evaluate(self, data: ProcessedData) -> float:
        """Évalue l'individu sur les données fournies.
        
        Args:
            data: Données pour l'évaluation
            
        Returns:
            Score de fitness
        """
        try:
            # Exécution du backtest
            backtest_results = await self.chromosome.backtest(
                symbol=data.metadata['symbol'],
                start_time=data.metadata['start_time'],
                end_time=data.metadata['end_time']
            )
            
            # Mise à jour des statistiques
            self._update_stats(backtest_results)
            
            return self.stats.fitness_score
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'evaluate', 'individual_id': self.id})
            return float('-inf')

    def _update_stats(self, backtest_results: Dict[str, float]) -> None:
        """Met à jour les statistiques de l'individu."""
        try:
            # Mise à jour du score de fitness
            self.stats.fitness_score = backtest_results.get('fitness_score', 0.0)
            
            # Mise à jour de l'historique des performances
            self.stats.performance_history.append(self.stats.fitness_score)
            self.stats.avg_performance = np.mean(self.stats.performance_history)
            self.stats.best_performance = max(
                self.stats.best_performance,
                self.stats.fitness_score
            )
            self.stats.worst_performance = min(
                self.stats.worst_performance,
                self.stats.fitness_score
            )
            
            # Création de l'historique des trades
            self.stats.trade_history = TradeHistory(
                trades=backtest_results.get('trades', []),
                total_trades=backtest_results.get('total_trades', 0),
                winning_trades=backtest_results.get('winning_trades', 0),
                losing_trades=backtest_results.get('losing_trades', 0),
                profit_factor=backtest_results.get('profit_factor', 0.0),
                max_drawdown=backtest_results.get('max_drawdown', 0.0),
                sharpe_ratio=backtest_results.get('sharpe_ratio', 0.0),
                win_rate=backtest_results.get('win_rate', 0.0)
            )
            
            # Mise à jour du nombre total de trades
            self.stats.lifetime_trades += self.stats.trade_history.total_trades
            
        except Exception as e:
            self.logger.log_error(e, {'action': '_update_stats', 'individual_id': self.id})

    def mutate(self, mutation_rate: float, method: str = 'gaussian') -> None:
        """Applique une mutation à l'individu.
        
        Args:
            mutation_rate: Taux de mutation
            method: Méthode de mutation
        """
        try:
            self.chromosome.mutate(mutation_rate, method)
            self.metadata['mutation_history'].append({
                'timestamp': datetime.now().isoformat(),
                'rate': mutation_rate,
                'method': method
            })
        except Exception as e:
            self.logger.log_error(e, {'action': 'mutate', 'individual_id': self.id})

    def crossover(self, other: 'TradingIndividual',
                 method: str = 'uniform') -> Tuple['TradingIndividual', 'TradingIndividual']:
        """Effectue un croisement avec un autre individu.
        
        Args:
            other: Autre individu pour le croisement
            method: Méthode de croisement
            
        Returns:
            Tuple des deux enfants
        """
        try:
            # Croisement des chromosomes
            child1_chromosome, child2_chromosome = self.chromosome.crossover(
                other.chromosome,
                method=method
            )
            
            # Création des nouveaux individus
            child1 = TradingIndividual(child1_chromosome)
            child2 = TradingIndividual(child2_chromosome)
            
            # Mise à jour des métadonnées
            crossover_info = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'parent_ids': [self.id, other.id]
            }
            
            for child in [child1, child2]:
                child.metadata['parent_ids'] = [self.id, other.id]
                child.metadata['crossover_history'].append(crossover_info)
                child.stats.generation_created = max(
                    self.stats.generation_created,
                    other.stats.generation_created
                ) + 1
            
            return child1, child2
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'crossover', 'individual_id': self.id})
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'individu."""
        return {
            'id': self.id,
            'fitness_score': self.stats.fitness_score,
            'generation': self.stats.generation_created,
            'lifetime_trades': self.stats.lifetime_trades,
            'performance': {
                'best': self.stats.best_performance,
                'worst': self.stats.worst_performance,
                'average': self.stats.avg_performance
            },
            'trade_history': {
                'total_trades': self.stats.trade_history.total_trades if self.stats.trade_history else 0,
                'win_rate': self.stats.trade_history.win_rate if self.stats.trade_history else 0.0,
                'profit_factor': self.stats.trade_history.profit_factor if self.stats.trade_history else 0.0
            },
            'metadata': self.metadata
        }

    def clone(self) -> 'TradingIndividual':
        """Crée une copie profonde de l'individu."""
        try:
            clone = TradingIndividual()
            clone.chromosome = self.chromosome  # Le chromosome a sa propre méthode de copie
            clone.stats = IndividualStats(
                fitness_score=self.stats.fitness_score,
                generation_created=self.stats.generation_created,
                lifetime_trades=self.stats.lifetime_trades,
                best_performance=self.stats.best_performance,
                worst_performance=self.stats.worst_performance,
                avg_performance=self.stats.avg_performance,
                performance_history=self.stats.performance_history.copy()
            )
            clone.metadata = {
                'created_at': datetime.now().isoformat(),
                'cloned_from': self.id,
                'original_created_at': self.metadata['created_at'],
                'parent_ids': self.metadata['parent_ids'].copy(),
                'mutation_history': self.metadata['mutation_history'].copy(),
                'crossover_history': self.metadata['crossover_history'].copy()
            }
            return clone
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'clone', 'individual_id': self.id})
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création d'un individu
        individual = TradingIndividual()
        
        # Test d'évaluation
        data = ProcessedData(...)  # À compléter
        fitness = await individual.evaluate(data)
        
        # Test de mutation
        individual.mutate(mutation_rate=0.1)
        
        # Test de croisement
        other = TradingIndividual()
        child1, child2 = individual.crossover(other)
        
        # Affichage des résultats
        print("Résumé de l'individu:")
        print(individual.get_summary())
        
        print("\nRésumé de l'enfant:")
        print(child1.get_summary())

    # Exécution
    import asyncio
    asyncio.run(main())