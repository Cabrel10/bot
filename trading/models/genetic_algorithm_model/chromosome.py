from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

from ...utils.logger import TradingLogger
from ...data.data_types import FeatureSet
from trading.core.data import HistoricalDataManager
from .params import GeneticAlgorithmParams

@dataclass
class GeneSegment:
    """Segment de gènes représentant une partie spécifique de la stratégie."""
    name: str
    values: np.ndarray
    length: int
    value_range: Tuple[float, float]

class TradingChromosome:
    """Représentation génétique d'une stratégie de trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise un chromosome avec une configuration optionnelle.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.segments: Dict[str, GeneSegment] = {}
        self.fitness_score: float = 0.0
        self.metadata: Dict[str, Any] = {
            'generation': 0,
            'parent_ids': [],
            'mutation_history': [],
            'creation_timestamp': None
        }
        self.data_manager = HistoricalDataManager()
        self.params = GeneticAlgorithmParams(config_path)
        self._initialize_segments()

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Charge la configuration du chromosome."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['chromosome']

    def _initialize_segments(self) -> None:
        """Initialise les segments de gènes selon la configuration."""
        try:
            for gene_config in self.config['gene_structure']:
                values = self._generate_random_values(
                    length=gene_config['length'],
                    value_range=tuple(gene_config['range'])
                )
                
                self.segments[gene_config['name']] = GeneSegment(
                    name=gene_config['name'],
                    values=values,
                    length=gene_config['length'],
                    value_range=tuple(gene_config['range'])
                )

        except Exception as e:
            self.logger.log_error(e, {'action': 'initialize_segments'})
            raise

    def _generate_random_values(self, 
                              length: int, 
                              value_range: Tuple[float, float]) -> np.ndarray:
        """Génère des valeurs aléatoires pour un segment."""
        return np.random.uniform(
            low=value_range[0],
            high=value_range[1],
            size=length
        )

    def normalize_weights(self, segment_name: str) -> None:
        """Normalise les poids d'un segment pour que leur somme soit égale à 1."""
        if segment_name in self.segments:
            segment = self.segments[segment_name]
            total = np.sum(np.abs(segment.values))
            if total > 0:
                segment.values = segment.values / total

    def mutate(self, mutation_rate: float, method: str = 'gaussian') -> None:
        """Applique une mutation au chromosome.
        
        Args:
            mutation_rate: Probabilité de mutation pour chaque gène
            method: Méthode de mutation ('gaussian', 'uniform', 'swap')
        """
        try:
            for segment in self.segments.values():
                if method == 'gaussian':
                    # Mutation gaussienne
                    mask = np.random.random(segment.length) < mutation_rate
                    noise = np.random.normal(0, 0.1, segment.length)
                    segment.values[mask] += noise[mask]
                    
                elif method == 'uniform':
                    # Mutation uniforme
                    mask = np.random.random(segment.length) < mutation_rate
                    segment.values[mask] = np.random.uniform(
                        segment.value_range[0],
                        segment.value_range[1],
                        np.sum(mask)
                    )
                    
                elif method == 'swap':
                    # Mutation par échange
                    if np.random.random() < mutation_rate:
                        idx1, idx2 = np.random.choice(segment.length, 2, replace=False)
                        segment.values[idx1], segment.values[idx2] = \
                            segment.values[idx2], segment.values[idx1]

                # Clip values to valid range
                segment.values = np.clip(
                    segment.values,
                    segment.value_range[0],
                    segment.value_range[1]
                )

            self.metadata['mutation_history'].append({
                'generation': self.metadata['generation'],
                'method': method,
                'rate': mutation_rate
            })

        except Exception as e:
            self.logger.log_error(e, {'action': 'mutate', 'method': method})
            raise

    def crossover(self, other: 'TradingChromosome', 
                 method: str = 'uniform') -> Tuple['TradingChromosome', 'TradingChromosome']:
        """Effectue un croisement avec un autre chromosome.
        
        Args:
            other: Autre chromosome pour le croisement
            method: Méthode de croisement ('uniform', 'single_point', 'arithmetic')
            
        Returns:
            Tuple de deux nouveaux chromosomes
        """
        try:
            child1 = TradingChromosome()
            child2 = TradingChromosome()
            
            for segment_name, segment in self.segments.items():
                if method == 'uniform':
                    # Croisement uniforme
                    mask = np.random.random(segment.length) < 0.5
                    child1.segments[segment_name].values = np.where(
                        mask, segment.values, other.segments[segment_name].values
                    )
                    child2.segments[segment_name].values = np.where(
                        mask, other.segments[segment_name].values, segment.values
                    )
                    
                elif method == 'single_point':
                    # Croisement à un point
                    point = np.random.randint(1, segment.length)
                    child1.segments[segment_name].values = np.concatenate([
                        segment.values[:point],
                        other.segments[segment_name].values[point:]
                    ])
                    child2.segments[segment_name].values = np.concatenate([
                        other.segments[segment_name].values[:point],
                        segment.values[point:]
                    ])
                    
                elif method == 'arithmetic':
                    # Croisement arithmétique
                    alpha = np.random.random()
                    child1.segments[segment_name].values = alpha * segment.values + \
                        (1 - alpha) * other.segments[segment_name].values
                    child2.segments[segment_name].values = (1 - alpha) * segment.values + \
                        alpha * other.segments[segment_name].values

            # Mise à jour des métadonnées
            for child in [child1, child2]:
                child.metadata['parent_ids'] = [id(self), id(other)]
                child.metadata['generation'] = max(
                    self.metadata['generation'],
                    other.metadata['generation']
                ) + 1

            return child1, child2

        except Exception as e:
            self.logger.log_error(e, {'action': 'crossover', 'method': method})
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le chromosome en dictionnaire."""
        return {
            'segments': {
                name: {
                    'values': segment.values.tolist(),
                    'length': segment.length,
                    'range': segment.value_range
                }
                for name, segment in self.segments.items()
            },
            'fitness_score': self.fitness_score,
            'metadata': self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Charge le chromosome depuis un dictionnaire."""
        try:
            for name, segment_data in data['segments'].items():
                self.segments[name] = GeneSegment(
                    name=name,
                    values=np.array(segment_data['values']),
                    length=segment_data['length'],
                    value_range=tuple(segment_data['range'])
                )
            self.fitness_score = data['fitness_score']
            self.metadata = data['metadata']
        except Exception as e:
            self.logger.log_error(e, {'action': 'from_dict'})
            raise

    def validate(self) -> bool:
        """Valide la structure et les valeurs du chromosome."""
        try:
            for segment in self.segments.values():
                # Vérification de la longueur
                if len(segment.values) != segment.length:
                    return False
                
                # Vérification des bornes
                if np.any(segment.values < segment.value_range[0]) or \
                   np.any(segment.values > segment.value_range[1]):
                    return False
                
                # Vérification des NaN
                if np.any(np.isnan(segment.values)):
                    return False

            return True

        except Exception as e:
            self.logger.log_error(e, {'action': 'validate'})
            return False

    async def backtest(self, 
                      symbol: str,
                      start_time: datetime,
                      end_time: datetime) -> Dict[str, float]:
        """Effectue un backtest de la stratégie."""
        try:
            # Récupération des données
            data = await self.data_manager.fetch_ohlcv(
                symbol=symbol,
                timeframe='1h',  # À configurer
                start_time=start_time,
                end_time=end_time
            )
            
            # Application de la stratégie
            trading_params = self.get_trading_parameters()
            # Logique de backtest à implémenter
            
            return {
                'pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }

        except Exception as e:
            self.logger.log_error(e, {'action': 'backtest'})
            raise

    def get_trading_parameters(self) -> Dict[str, Any]:
        """Convertit les segments en paramètres de trading utilisables."""
        try:
            return {
                'entry_conditions': self._convert_entry_weights(),
                'exit_conditions': self._convert_exit_weights(),
                'timeframe_weights': self._convert_timeframe_weights(),
                'risk_parameters': self._convert_risk_params()
            }
        except Exception as e:
            self.logger.log_error(e, {'action': 'get_trading_parameters'})
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un chromosome
    chromosome = TradingChromosome()
    
    # Affichage des segments
    print("Segments du chromosome:")
    for name, segment in chromosome.segments.items():
        print(f"{name}: {segment.values}")
    
    # Test de mutation
    chromosome.mutate(mutation_rate=0.1, method='gaussian')
    
    # Test de croisement
    other_chromosome = TradingChromosome()
    child1, child2 = chromosome.crossover(other_chromosome, method='uniform')
    
    # Validation
    print("\nValidation des chromosomes:")
    print(f"Original: {chromosome.validate()}")
    print(f"Child 1: {child1.validate()}")
    print(f"Child 2: {child2.validate()}")