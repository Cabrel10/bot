"""
Métriques de performance avancées pour le modèle hybride.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats
import tensorflow as tf
from tensorflow.keras import backend as K

@dataclass
class MetricsConfig:
    """Configuration pour les métriques avancées."""
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    rolling_window: int = 252
    min_trades: int = 10
    max_drawdown_threshold: float = 0.2

class AdvancedMetrics:
    """Calcul des métriques de performance avancées."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure la journalisation."""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def calculate_returns_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcule les métriques de rendement avancées.
        
        Args:
            returns: Rendements du portefeuille
            benchmark_returns: Rendements du benchmark (optionnel)
            
        Returns:
            Dictionnaire des métriques
        """
        try:
            metrics = {}
            
            # Rendement total
            metrics['total_return'] = np.prod(1 + returns) - 1
            
            # Rendement annualisé
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (252/len(returns)) - 1
            
            # Volatilité
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            
            # Ratio de Sharpe
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            
            # Ratio de Sortino
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            
            # Ratio de Calmar
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)
            
            # Ratio d'Omega
            metrics['omega_ratio'] = self._calculate_omega_ratio(returns)
            
            # Value at Risk (VaR)
            metrics['var_95'] = self._calculate_var(returns)
            
            # Conditional VaR
            metrics['cvar_95'] = self._calculate_cvar(returns)
            
            # Maximum Drawdown
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            
            # Taux de réussite
            metrics['win_rate'] = np.mean(returns > 0)
            
            # Ratio de profit
            metrics['profit_factor'] = self._calculate_profit_factor(returns)
            
            # Métriques relatives au benchmark
            if benchmark_returns is not None:
                metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de rendement: {e}")
            raise
    
    def calculate_trade_metrics(
        self,
        trades: List[Dict],
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calcule les métriques liées aux trades.
        
        Args:
            trades: Liste des trades
            initial_capital: Capital initial
            
        Returns:
            Dictionnaire des métriques
        """
        try:
            if len(trades) < self.config.min_trades:
                self.logger.warning(
                    f"Pas assez de trades pour l'analyse: {len(trades)} < {self.config.min_trades}"
                )
                return {}
            
            metrics = {}
            
            # Nombre total de trades
            metrics['total_trades'] = len(trades)
            
            # Trades gagnants et perdants
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            # Taux de réussite
            metrics['win_rate'] = len(winning_trades) / len(trades)
            
            # Profit moyen par trade gagnant
            metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            
            # Perte moyenne par trade perdant
            metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Ratio de profit
            metrics['profit_factor'] = (
                abs(sum(t['pnl'] for t in winning_trades)) /
                abs(sum(t['pnl'] for t in losing_trades))
                if losing_trades else float('inf')
            )
            
            # Retour sur investissement (ROI)
            metrics['roi'] = sum(t['pnl'] for t in trades) / initial_capital
            
            # Trades consécutifs
            metrics['max_consecutive_wins'] = self._calculate_max_consecutive_trades(winning_trades)
            metrics['max_consecutive_losses'] = self._calculate_max_consecutive_trades(losing_trades)
            
            # Durée moyenne des trades
            metrics['avg_trade_duration'] = np.mean([
                (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                for t in trades
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de trades: {e}")
            raise
    
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcule les métriques de risque avancées.
        
        Args:
            returns: Rendements
            positions: Positions du portefeuille
            
        Returns:
            Dictionnaire des métriques
        """
        try:
            metrics = {}
            
            # Beta
            metrics['beta'] = self._calculate_beta(returns)
            
            # Alpha
            metrics['alpha'] = self._calculate_alpha(returns)
            
            # Ratio d'information
            metrics['information_ratio'] = self._calculate_information_ratio(returns)
            
            # Ratio de Treynor
            metrics['treynor_ratio'] = self._calculate_treynor_ratio(returns)
            
            # Ratio de Jensen
            metrics['jensen_ratio'] = self._calculate_jensen_ratio(returns)
            
            # Ratio de Modigliani
            metrics['modigliani_ratio'] = self._calculate_modigliani_ratio(returns)
            
            # Ratio de Sortino modifié
            metrics['modified_sortino_ratio'] = self._calculate_modified_sortino_ratio(returns)
            
            # Ratio de Calmar modifié
            metrics['modified_calmar_ratio'] = self._calculate_modified_calmar_ratio(returns)
            
            # Ratio d'Omega modifié
            metrics['modified_omega_ratio'] = self._calculate_modified_omega_ratio(returns)
            
            # Ratio de Sterling
            metrics['sterling_ratio'] = self._calculate_sterling_ratio(returns)
            
            # Ratio de Burke
            metrics['burke_ratio'] = self._calculate_burke_ratio(returns)
            
            # Ratio de Martin
            metrics['martin_ratio'] = self._calculate_martin_ratio(returns)
            
            # Ratio de Pain
            metrics['pain_ratio'] = self._calculate_pain_ratio(returns)
            
            # Ratio de Kappa
            metrics['kappa_ratio'] = self._calculate_kappa_ratio(returns)
            
            # Ratio de Upside Potential
            metrics['upside_potential_ratio'] = self._calculate_upside_potential_ratio(returns)
            
            # Ratio de Calmar modifié
            metrics['modified_calmar_ratio'] = self._calculate_modified_calmar_ratio(returns)
            
            # Ratio de Sortino modifié
            metrics['modified_sortino_ratio'] = self._calculate_modified_sortino_ratio(returns)
            
            # Ratio d'Omega modifié
            metrics['modified_omega_ratio'] = self._calculate_modified_omega_ratio(returns)
            
            # Ratio de Sterling
            metrics['sterling_ratio'] = self._calculate_sterling_ratio(returns)
            
            # Ratio de Burke
            metrics['burke_ratio'] = self._calculate_burke_ratio(returns)
            
            # Ratio de Martin
            metrics['martin_ratio'] = self._calculate_martin_ratio(returns)
            
            # Ratio de Pain
            metrics['pain_ratio'] = self._calculate_pain_ratio(returns)
            
            # Ratio de Kappa
            metrics['kappa_ratio'] = self._calculate_kappa_ratio(returns)
            
            # Ratio de Upside Potential
            metrics['upside_potential_ratio'] = self._calculate_upside_potential_ratio(returns)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de risque: {e}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sharpe."""
        excess_returns = returns - self.config.risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sortino."""
        excess_returns = returns - self.config.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Calmar."""
        annualized_return = (1 + np.prod(1 + returns)) ** (252/len(returns)) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    def _calculate_omega_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio d'Omega."""
        excess_returns = returns - self.config.risk_free_rate/252
        positive_returns = np.sum(excess_returns[excess_returns > 0])
        negative_returns = abs(np.sum(excess_returns[excess_returns < 0]))
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calcule la Value at Risk."""
        return np.percentile(returns, (1 - self.config.confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray) -> float:
        """Calcule la Conditional Value at Risk."""
        var = self._calculate_var(returns)
        return np.mean(returns[returns <= var])
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcule le maximum drawdown."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calcule le ratio de profit."""
        positive_returns = np.sum(returns[returns > 0])
        negative_returns = abs(np.sum(returns[returns < 0]))
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    def _calculate_benchmark_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calcule les métriques relatives au benchmark."""
        metrics = {}
        
        # Beta
        covariance = np.cov(returns, benchmark_returns)[0,1]
        benchmark_variance = np.var(benchmark_returns)
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Alpha
        metrics['alpha'] = np.mean(returns) - metrics['beta'] * np.mean(benchmark_returns)
        
        # Tracking Error
        metrics['tracking_error'] = np.std(returns - benchmark_returns) * np.sqrt(252)
        
        # Information Ratio
        excess_returns = returns - benchmark_returns
        metrics['information_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return metrics
    
    def _calculate_max_consecutive_trades(self, trades: List[Dict]) -> int:
        """Calcule le nombre maximum de trades consécutifs."""
        if not trades:
            return 0
        
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(trades)):
            if (trades[i]['entry_time'] - trades[i-1]['exit_time']).total_seconds() < 3600:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calcule le beta."""
        return np.cov(returns, np.ones_like(returns))[0,1] / np.var(returns)
    
    def _calculate_alpha(self, returns: np.ndarray) -> float:
        """Calcule l'alpha."""
        return np.mean(returns) - self.config.risk_free_rate/252
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio d'information."""
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_treynor_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Treynor."""
        beta = self._calculate_beta(returns)
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / beta * np.sqrt(252) if beta != 0 else float('inf')
    
    def _calculate_jensen_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Jensen."""
        beta = self._calculate_beta(returns)
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) - beta * np.mean(excess_returns)
    
    def _calculate_modigliani_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Modigliani."""
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_modified_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sortino modifié."""
        excess_returns = returns - self.config.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_modified_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Calmar modifié."""
        annualized_return = (1 + np.prod(1 + returns)) ** (252/len(returns)) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    def _calculate_modified_omega_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio d'Omega modifié."""
        excess_returns = returns - self.config.risk_free_rate/252
        positive_returns = np.sum(excess_returns[excess_returns > 0])
        negative_returns = abs(np.sum(excess_returns[excess_returns < 0]))
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    def _calculate_sterling_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sterling."""
        annualized_return = (1 + np.prod(1 + returns)) ** (252/len(returns)) - 1
        max_drawdown = self._calculate_max_drawdown(returns)
        return annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    def _calculate_burke_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Burke."""
        annualized_return = (1 + np.prod(1 + returns)) ** (252/len(returns)) - 1
        drawdowns = self._calculate_drawdowns(returns)
        return annualized_return / np.sqrt(np.sum(drawdowns**2)) if len(drawdowns) > 0 else float('inf')
    
    def _calculate_martin_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Martin."""
        annualized_return = (1 + np.prod(1 + returns)) ** (252/len(returns)) - 1
        drawdowns = self._calculate_drawdowns(returns)
        return annualized_return / np.sqrt(np.sum(drawdowns**2)) if len(drawdowns) > 0 else float('inf')
    
    def _calculate_pain_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Pain."""
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_kappa_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Kappa."""
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_upside_potential_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Upside Potential."""
        excess_returns = returns - self.config.risk_free_rate/252
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        return np.mean(positive_returns) / np.std(negative_returns) if len(negative_returns) > 0 else float('inf')
    
    def _calculate_drawdowns(self, returns: np.ndarray) -> np.ndarray:
        """Calcule les drawdowns."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns[drawdowns < 0] 