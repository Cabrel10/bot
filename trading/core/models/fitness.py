"""
Module d'évaluation du fitness pour les stratégies de trading.

Ce module fournit des classes et des fonctions pour évaluer la performance
des stratégies de trading en utilisant diverses métriques financières.
Il supporte l'évaluation multi-objectif et la gestion des contrats futures.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from enum import Enum
import logging

# Importations corrigées pour correspondre à la nouvelle structure
from src.core.position import Position
from src.services.backtesting.performance_analyzer import calculate_metrics
from src.core.exceptions import ValidationError, CalculationError

# Configuration du logging
logger = logging.getLogger(__name__)

class MarketType(Enum):
    """Types de marchés supportés."""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"

@dataclass
class FitnessConfig:
    """Configuration pour l'évaluation du fitness.
    
    Attributes:
        weights (Dict[str, float]): Poids des différentes métriques dans le score final.
            Doit sommer à 1.
        constraints (Dict[str, Union[float, int]]): Contraintes pour l'évaluation.
            Ex: min_trades, max_drawdown, etc.
        objectives (List[str]): Liste des objectifs pour l'optimisation multi-objectif.
        penalty_factor (float): Facteur de pénalité pour les violations de contraintes.
        market_type (MarketType): Type de marché (spot, futures, margin).
        futures_config (Dict[str, Any]): Configuration spécifique aux futures.
        volume_config (Dict[str, Any]): Configuration des indicateurs de volume.
    """
    weights: Dict[str, float]
    constraints: Dict[str, Union[float, int]]
    objectives: List[str]
    penalty_factor: float = 1.0
    market_type: MarketType = MarketType.SPOT
    futures_config: Dict[str, Any] = field(default_factory=dict)
    volume_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation post-initialisation."""
        self._validate_weights()
        self._validate_constraints()
        self._validate_futures_config()
        self._validate_volume_config()

    def _validate_weights(self):
        """Valide les poids des métriques."""
        if not self.weights:
            raise ValidationError("Les poids ne peuvent pas être vides")
        
        if not all(isinstance(w, (int, float)) for w in self.weights.values()):
            raise ValidationError("Tous les poids doivent être numériques")
        
        if not all(w >= 0 for w in self.weights.values()):
            raise ValidationError("Tous les poids doivent être positifs")
        
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValidationError(f"La somme des poids doit être 1, trouvé: {total}")

    def _validate_constraints(self):
        """Valide les contraintes."""
        required_constraints = {'min_trades', 'max_drawdown', 'min_profit'}
        missing = required_constraints - set(self.constraints.keys())
        if missing:
            raise ValidationError(f"Contraintes manquantes: {missing}")

        if self.constraints['min_trades'] < 1:
            raise ValidationError("min_trades doit être >= 1")
        
        if not (-1 < self.constraints['max_drawdown'] < 0):
            raise ValidationError("max_drawdown doit être entre -1 et 0")

    def _validate_futures_config(self):
        """Valide la configuration des futures si activée."""
        if self.market_type == MarketType.FUTURES:
            required_fields = {'leverage', 'funding_rate', 'margin_type'}
            missing = required_fields - set(self.futures_config.keys())
            if missing:
                raise ValidationError(f"Configuration futures manquante: {missing}")
            
            if not (0 < self.futures_config['leverage'] <= 100):
                raise ValidationError("Le levier doit être entre 0 et 100")

    def _validate_volume_config(self):
        """Valide la configuration des indicateurs de volume si activés."""
        if self.volume_config:
            required_fields = {'vwap', 'volume_profile'}
            missing = required_fields - set(self.volume_config.keys())
            if missing:
                raise ValidationError(f"Configuration volume manquante: {missing}")

@dataclass
class FitnessMetrics:
    """Métriques de performance pour l'évaluation du fitness.
    
    Attributes:
        sharpe_ratio (float): Ratio de Sharpe annualisé.
        sortino_ratio (float): Ratio de Sortino annualisé.
        max_drawdown (float): Drawdown maximum en pourcentage.
        win_rate (float): Taux de trades gagnants.
        profit_factor (float): Ratio gains/pertes.
        total_return (float): Rendement total en pourcentage.
        volatility (float): Volatilité annualisée.
        calmar_ratio (float): Ratio de Calmar.
        futures_metrics (Dict[str, float]): Métriques spécifiques aux futures.
        volume_metrics (Dict[str, float]): Métriques basées sur le volume.
    """
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    futures_metrics: Dict[str, float] = field(default_factory=dict)
    volume_metrics: Dict[str, float] = field(default_factory=dict)

class FitnessEvaluator:
    """Évaluateur de fitness pour les modèles de trading.
    
    Cette classe fournit des méthodes pour évaluer la performance des stratégies
    de trading en utilisant diverses métriques financières. Elle supporte
    l'évaluation multi-objectif et la gestion des contrats futures.
    """

    def __init__(self, config: FitnessConfig):
        """
        Initialise l'évaluateur de fitness.
        
        Args:
            config: Configuration de l'évaluation du fitness.
            
        Raises:
            ValidationError: Si la configuration est invalide.
        """
        self.config = config
        self.risk_free_rate = 0.02  # Taux sans risque annuel
        
        # Validation de la configuration
        if not isinstance(config, FitnessConfig):
            raise ValidationError("config doit être une instance de FitnessConfig")

    def calculate_fitness(self, 
                         predictions: np.ndarray,
                         actual_returns: np.ndarray,
                         positions: Optional[np.ndarray] = None,
                         additional_data: Optional[Dict[str, Any]] = None) -> FitnessMetrics:
        """
        Calcule le score de fitness global basé sur plusieurs métriques.
        
        Args:
            predictions: Prédictions du modèle
            actual_returns: Rendements réels
            positions: Positions prises (optionnel)
            additional_data: Données supplémentaires (futures, volume, etc.)
        
        Returns:
            FitnessMetrics: Ensemble des métriques de performance
            
        Raises:
            ValidationError: Si les données d'entrée sont invalides
            CalculationError: Si une erreur survient pendant les calculs
        """
        try:
            # Validation des entrées
            self._validate_inputs(predictions, actual_returns, positions)
            
            # Calcul des rendements du modèle
            model_returns = self._calculate_model_returns(predictions, actual_returns, positions)
            
            # Calcul des métriques de base
            metrics = FitnessMetrics(
                sharpe_ratio=self._calculate_sharpe_ratio(model_returns),
                sortino_ratio=self._calculate_sortino_ratio(model_returns),
                max_drawdown=self._calculate_max_drawdown(model_returns),
                win_rate=self._calculate_win_rate(model_returns),
                profit_factor=self._calculate_profit_factor(model_returns),
                total_return=self._calculate_total_return(model_returns),
                volatility=self._calculate_volatility(model_returns),
                calmar_ratio=self._calculate_calmar_ratio(model_returns)
            )
            
            # Ajout des métriques futures si nécessaire
            if self.config.market_type == MarketType.FUTURES and additional_data:
                metrics.futures_metrics = self._calculate_futures_metrics(
                    model_returns, additional_data
                )
            
            # Ajout des métriques de volume si configurées
            if self.config.volume_config and additional_data:
                metrics.volume_metrics = self._calculate_volume_metrics(
                    model_returns, additional_data
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du fitness: {str(e)}")
            raise CalculationError(f"Erreur lors du calcul du fitness: {str(e)}")

    def _validate_inputs(self,
                        predictions: np.ndarray,
                        actual_returns: np.ndarray,
                        positions: Optional[np.ndarray] = None) -> None:
        """
        Valide les données d'entrée pour le calcul du fitness.
        
        Args:
            predictions: Prédictions du modèle
            actual_returns: Rendements réels
            positions: Positions prises (optionnel)
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        if not isinstance(predictions, np.ndarray):
            raise ValidationError("predictions doit être un numpy.ndarray")
            
        if not isinstance(actual_returns, np.ndarray):
            raise ValidationError("actual_returns doit être un numpy.ndarray")
            
        if positions is not None and not isinstance(positions, np.ndarray):
            raise ValidationError("positions doit être un numpy.ndarray")
            
        if len(predictions) != len(actual_returns):
            raise ValidationError("predictions et actual_returns doivent avoir la même taille")
            
        if positions is not None and len(positions) != len(predictions):
            raise ValidationError("positions doit avoir la même taille que predictions")
            
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValidationError("predictions contient des valeurs nan ou inf")
            
        if np.any(np.isnan(actual_returns)) or np.any(np.isinf(actual_returns)):
            raise ValidationError("actual_returns contient des valeurs nan ou inf")

    def _calculate_futures_metrics(self,
                                 returns: np.ndarray,
                                 additional_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcule les métriques spécifiques aux contrats futures.
        
        Args:
            returns: Rendements du modèle
            additional_data: Données supplémentaires (funding rate, etc.)
            
        Returns:
            Dict[str, float]: Métriques futures calculées
        """
        try:
            funding_rate = additional_data.get('funding_rate')
            if funding_rate is None:
                raise ValidationError("Données de funding rate manquantes")
                
            # Calcul du coût de funding
            funding_cost = np.sum(funding_rate * returns)
            
            # Calcul du rendement ajusté au levier
            leverage = self.config.futures_config['leverage']
            leveraged_returns = returns * leverage
            
            # Calcul des métriques
            return {
                'funding_cost': funding_cost,
                'leveraged_sharpe': self._calculate_sharpe_ratio(leveraged_returns),
                'leveraged_sortino': self._calculate_sortino_ratio(leveraged_returns),
                'leveraged_drawdown': self._calculate_max_drawdown(leveraged_returns)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques futures: {str(e)}")
            raise CalculationError(f"Erreur lors du calcul des métriques futures: {str(e)}")

    def _calculate_volume_metrics(self,
                                returns: np.ndarray,
                                additional_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcule les métriques basées sur le volume.
        
        Args:
            returns: Rendements du modèle
            additional_data: Données supplémentaires (volume, VWAP, etc.)
            
        Returns:
            Dict[str, float]: Métriques de volume calculées
        """
        try:
            volume = additional_data.get('volume')
            vwap = additional_data.get('vwap')
            if volume is None or vwap is None:
                raise ValidationError("Données de volume manquantes")
                
            # Calcul des métriques de volume
            volume_ma = pd.Series(volume).rolling(
                window=self.config.volume_config.get('ma_window', 20)
            ).mean()
            
            relative_volume = volume / volume_ma
            
            return {
                'avg_relative_volume': np.mean(relative_volume),
                'volume_trend': np.corrcoef(returns, relative_volume)[0, 1],
                'vwap_efficiency': np.mean(np.abs(returns[returns * (vwap - np.mean(vwap)) > 0]))
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques de volume: {str(e)}")
            raise CalculationError(f"Erreur lors du calcul des métriques de volume: {str(e)}")

    def evaluate_population(self, 
                          population_metrics: List[Dict[str, float]],
                          weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Évalue une population entière.
        
        Args:
            population_metrics: Liste des métriques pour chaque individu
            weights: Poids optionnels pour les métriques
            
        Returns:
            np.ndarray: Array des scores de fitness
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        if not population_metrics:
            raise ValidationError("population_metrics ne peut pas être vide")
            
        weights = weights or self.config.weights
        
        try:
            scores = []
            for metrics in population_metrics:
                # Calcul du score pondéré
                score = sum(
                    weights.get(metric, 0) * value
                    for metric, value in metrics.items()
                    if metric in weights
                )
                
                # Application des pénalités
                if metrics.get('max_drawdown', 0) < self.config.constraints['max_drawdown']:
                    score *= (1 - self.config.penalty_factor)
                    
                if metrics.get('total_trades', 0) < self.config.constraints['min_trades']:
                    score *= (1 - self.config.penalty_factor)
                
                scores.append(score)
                
            return np.array(scores)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la population: {str(e)}")
            raise CalculationError(f"Erreur lors de l'évaluation de la population: {str(e)}")

    def get_pareto_front(self, 
                        population_metrics: List[Dict[str, float]]) -> Tuple[List[int], np.ndarray]:
        """
        Identifie le front de Pareto pour l'optimisation multi-objectif.
        
        Args:
            population_metrics: Liste des métriques pour chaque individu
            
        Returns:
            Tuple[List[int], np.ndarray]: Indices des solutions Pareto-optimales et leurs scores
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        if not population_metrics:
            raise ValidationError("population_metrics ne peut pas être vide")
            
        if not self.config.objectives:
            raise ValidationError("Les objectifs doivent être définis pour l'optimisation multi-objectif")
            
        try:
            n_individuals = len(population_metrics)
            pareto_front = np.ones(n_individuals, dtype=bool)
            
            # Identification des solutions non-dominées
            for i in range(n_individuals):
                for j in range(n_individuals):
                    if i != j:
                        dominates = True
                        for objective in self.config.objectives:
                            if population_metrics[i][objective] <= population_metrics[j][objective]:
                                dominates = False
                                break
                        if dominates:
                            pareto_front[j] = False
            
            pareto_indices = np.where(pareto_front)[0]
            pareto_scores = self.evaluate_population([
                population_metrics[i] for i in pareto_indices
            ])
            
            return list(pareto_indices), pareto_scores
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du front de Pareto: {str(e)}")
            raise CalculationError(f"Erreur lors du calcul du front de Pareto: {str(e)}")

    def evaluate(self, 
                positions: List[Position],
                initial_capital: float,
                market_data: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict]:
        """
        Évalue la performance d'une stratégie basée sur ses positions.
        
        Args:
            positions: Liste des positions fermées
            initial_capital: Capital initial utilisé
            market_data: Données de marché supplémentaires
            
        Returns:
            Tuple[float, Dict]: Score de fitness et métriques détaillées
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        if not positions:
            raise ValidationError("La liste des positions ne peut pas être vide")
            
        if initial_capital <= 0:
            raise ValidationError("Le capital initial doit être positif")
            
        try:
            # Validation du nombre minimum de trades
            if len(positions) < self.config.constraints['min_trades']:
                return 0.0, {'error': 'Insufficient trades'}
            
            # Calcul des métriques de performance
            metrics = calculate_metrics(positions, initial_capital)
            
            # Ajout des métriques futures si nécessaire
            if self.config.market_type == MarketType.FUTURES and market_data:
                futures_metrics = self._calculate_futures_metrics(
                    np.array([p.pnl for p in positions]),
                    market_data
                )
                metrics.update(futures_metrics)
            
            # Ajout des métriques de volume si configurées
            if self.config.volume_config and market_data:
                volume_metrics = self._calculate_volume_metrics(
                    np.array([p.pnl for p in positions]),
                    market_data
                )
                metrics.update(volume_metrics)
            
            # Calcul du score final
            score = self._calculate_final_score(metrics)
            
            return score, metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de la stratégie: {str(e)}")
            raise CalculationError(f"Erreur lors de l'évaluation de la stratégie: {str(e)}")

    def _calculate_final_score(self, metrics: Dict[str, float]) -> float:
        """
        Calcule le score final à partir des métriques.
        
        Args:
            metrics: Dictionnaire des métriques calculées
            
        Returns:
            float: Score final
        """
        try:
            # Normalisation des composants du score
            sharpe_score = max(0, metrics['sharpe_ratio']) / 3.0
            profit_score = max(0, metrics['total_profit_pct']) / 100.0
            drawdown_score = 1.0 - min(1.0, abs(metrics['max_drawdown']) / 50.0)
            
            # Calcul du score pondéré
            weights = self.config.weights
            score = (
                weights.get('sharpe_ratio', 0.3) * sharpe_score +
                weights.get('total_return', 0.4) * profit_score +
                weights.get('max_drawdown', 0.3) * drawdown_score
            )
            
            # Application des pénalités si nécessaire
            if metrics['max_drawdown'] < self.config.constraints['max_drawdown']:
                score *= (1 - self.config.penalty_factor)
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score final: {str(e)}")
            raise CalculationError(f"Erreur lors du calcul du score final: {str(e)}")

    def _calculate_model_returns(self, 
                               predictions: np.ndarray,
                               actual_returns: np.ndarray,
                               positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Calcule les rendements du modèle."""
        if positions is None:
            positions = np.sign(predictions)
        return positions * actual_returns

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sharpe."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252  # Annualisé
        if np.std(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sortino."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcule le drawdown maximum."""
        if len(returns) < self.config.constraints['min_trades']:
            return 1.0
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns))

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calcule le taux de trades gagnants."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        return np.mean(returns > 0)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calcule le facteur de profit."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return gains / losses if losses != 0 else 0.0

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calcule le rendement total."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        return np.prod(1 + returns) - 1

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calcule la volatilité annualisée."""
        if len(returns) < self.config.constraints['min_trades']:
            return float('inf')
        return np.std(returns) * np.sqrt(252)

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Calmar."""
        if len(returns) < self.config.constraints['min_trades']:
            return 0.0
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        return self._calculate_total_return(returns) / max_dd

    def validate_strategy(self, positions: List[Position], 
                        min_profit: float = 0.0,
                        max_drawdown: float = -0.3) -> bool:
        """
        Valide si une stratégie répond aux critères minimaux.
        
        Args:
            positions: Liste des positions fermées
            min_profit: Profit minimum requis (en %)
            max_drawdown: Drawdown maximum autorisé (en %)
            
        Returns:
            bool: True si la stratégie est valide
        """
        if len(positions) < self.config.constraints['min_trades']:
            return False

        metrics = calculate_metrics(positions, 10000)  # Capital arbitraire pour le calcul
        
        return (metrics['total_profit_pct'] >= min_profit and 
                metrics['max_drawdown'] >= max_drawdown) 