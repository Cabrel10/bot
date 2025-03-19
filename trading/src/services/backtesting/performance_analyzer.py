from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy import stats

from ..utils.logger import TradingLogger
from src.core.data_types import Trade

@dataclass
class PerformanceMetrics:
    """Métriques de performance d'une stratégie."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    num_trades: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    time_in_market: float
    recovery_factor: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

class PerformanceAnalyzer:
    """Analyse des performances des stratégies de trading."""
    
    def __init__(self):
        """Initialise l'analyseur de performances."""
        self.logger = TradingLogger()
        
    def analyze_trades(self,
                      trades: List[Trade],
                      equity_curve: List[Dict],
                      initial_capital: float,
                      benchmark_returns: Optional[pd.Series] = None,
                      risk_free_rate: float = 0.02) -> Dict:
        """
        Analyse une liste de trades et calcule les métriques de performance.
        
        Args:
            trades: Liste des trades
            equity_curve: Courbe d'équité
            initial_capital: Capital initial
            benchmark_returns: Rendements du benchmark
            risk_free_rate: Taux sans risque annuel
            
        Returns:
            Dictionnaire des métriques de performance
        """
        try:
            # Conversion de la courbe d'équité en DataFrame
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calcul des rendements
            returns = self._calculate_returns(equity_df['equity'])
            
            # Calcul des métriques de base
            total_return = (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital
            annualized_return = self._calculate_annualized_return(total_return, returns.index)
            
            # Métriques de risque
            volatility = returns.std() * np.sqrt(252)  # Annualisée
            downside_volatility = returns[returns < 0].std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
            
            # Métriques de trade
            trade_metrics = self._analyze_trade_metrics(trades)
            
            # Ratios de performance
            sharpe = self._calculate_sharpe_ratio(returns, risk_free_rate)
            sortino = self._calculate_sortino_ratio(returns, risk_free_rate)
            calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
            omega = self._calculate_omega_ratio(returns)
            
            # Métriques relatives au benchmark
            benchmark_metrics = {}
            if benchmark_returns is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(
                    returns,
                    benchmark_returns
                )
            
            # Construction de l'objet PerformanceMetrics
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_drawdown,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                avg_trade=trade_metrics['avg_trade'],
                avg_win=trade_metrics['avg_win'],
                avg_loss=trade_metrics['avg_loss'],
                num_trades=len(trades),
                max_consecutive_wins=trade_metrics['max_consecutive_wins'],
                max_consecutive_losses=trade_metrics['max_consecutive_losses'],
                time_in_market=trade_metrics['time_in_market'],
                recovery_factor=total_return / abs(max_drawdown) if max_drawdown != 0 else np.inf,
                calmar_ratio=calmar,
                omega_ratio=omega,
                information_ratio=benchmark_metrics.get('information_ratio'),
                alpha=benchmark_metrics.get('alpha'),
                beta=benchmark_metrics.get('beta')
            )
            
            # Construction du rapport complet
            return {
                'metrics': metrics.__dict__,
                'trade_analysis': trade_metrics,
                'risk_metrics': {
                    'volatility': volatility,
                    'downside_volatility': downside_volatility,
                    'var_95': self._calculate_var(returns, 0.95),
                    'var_99': self._calculate_var(returns, 0.99),
                    'expected_shortfall': self._calculate_expected_shortfall(returns, 0.95)
                },
                'returns_analysis': {
                    'daily_returns': returns.describe().to_dict(),
                    'monthly_returns': self._calculate_monthly_returns(returns),
                    'drawdown_periods': self._analyze_drawdown_periods(equity_df['equity'])
                },
                'benchmark_analysis': benchmark_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des performances: {str(e)}")
            raise
            
    def _calculate_returns(self, equity: pd.Series) -> pd.Series:
        """
        Calcule les rendements quotidiens.
        
        Args:
            equity: Série temporelle de l'équité
            
        Returns:
            Série des rendements
        """
        return equity.pct_change().fillna(0)
        
    def _calculate_annualized_return(self,
                                   total_return: float,
                                   index: pd.DatetimeIndex) -> float:
        """
        Calcule le rendement annualisé.
        
        Args:
            total_return: Rendement total
            index: Index temporel
            
        Returns:
            Rendement annualisé
        """
        years = (index[-1] - index[0]).days / 365.25
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """
        Calcule le drawdown maximum.
        
        Args:
            equity: Série temporelle de l'équité
            
        Returns:
            Drawdown maximum en pourcentage
        """
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        return drawdowns.min()
        
    def _analyze_trade_metrics(self, trades: List[Trade]) -> Dict:
        """
        Analyse les métriques des trades.
        
        Args:
            trades: Liste des trades
            
        Returns:
            Métriques des trades
        """
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_trade': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'time_in_market': 0
            }
            
        # Calcul des gains et pertes
        profits = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        
        # Métriques de base
        win_rate = len(profits) / len(trades)
        profit_factor = abs(sum(profits) / sum(losses)) if losses else np.inf
        avg_trade = np.mean([t.pnl for t in trades])
        avg_win = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Séquences de gains/pertes
        current_streak = 0
        max_wins = 0
        max_losses = 0
        
        for t in trades:
            if t.pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_wins = max(max_wins, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_losses = min(max_losses, current_streak)
                
        # Temps en position
        total_time = sum((t.exit_time - t.entry_time).total_seconds() for t in trades)
        total_period = (trades[-1].exit_time - trades[0].entry_time).total_seconds()
        time_in_market = total_time / total_period if total_period > 0 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': abs(max_losses),
            'time_in_market': time_in_market,
            'trade_distribution': {
                'profit_trades': len(profits),
                'loss_trades': len(losses),
                'breakeven_trades': len(trades) - len(profits) - len(losses)
            }
        }
        
    def _calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              risk_free_rate: float) -> float:
        """
        Calcule le ratio de Sharpe.
        
        Args:
            returns: Série des rendements
            risk_free_rate: Taux sans risque annuel
            
        Returns:
            Ratio de Sharpe
        """
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
    def _calculate_sortino_ratio(self,
                               returns: pd.Series,
                               risk_free_rate: float) -> float:
        """
        Calcule le ratio de Sortino.
        
        Args:
            returns: Série des rendements
            risk_free_rate: Taux sans risque annuel
            
        Returns:
            Ratio de Sortino
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """
        Calcule le ratio Omega.
        
        Args:
            returns: Série des rendements
            threshold: Seuil de rendement
            
        Returns:
            Ratio Omega
        """
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        return gains / losses if losses != 0 else np.inf
        
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """
        Calcule la Value at Risk.
        
        Args:
            returns: Série des rendements
            confidence: Niveau de confiance
            
        Returns:
            VaR
        """
        return np.percentile(returns, (1 - confidence) * 100)
        
    def _calculate_expected_shortfall(self,
                                    returns: pd.Series,
                                    confidence: float) -> float:
        """
        Calcule l'Expected Shortfall (CVaR).
        
        Args:
            returns: Série des rendements
            confidence: Niveau de confiance
            
        Returns:
            Expected Shortfall
        """
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
        
    def _calculate_monthly_returns(self, returns: pd.Series) -> Dict:
        """
        Calcule les statistiques des rendements mensuels.
        
        Args:
            returns: Série des rendements quotidiens
            
        Returns:
            Statistiques mensuelles
        """
        monthly_returns = (1 + returns).resample('M').prod() - 1
        return {
            'mean': monthly_returns.mean(),
            'std': monthly_returns.std(),
            'best': monthly_returns.max(),
            'worst': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum() / len(monthly_returns),
            'distribution': monthly_returns.describe().to_dict()
        }
        
    def _analyze_drawdown_periods(self, equity: pd.Series) -> List[Dict]:
        """
        Analyse les périodes de drawdown.
        
        Args:
            equity: Série temporelle de l'équité
            
        Returns:
            Liste des périodes de drawdown
        """
        drawdown_periods = []
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        
        # Identification des périodes
        in_drawdown = False
        start_idx = None
        max_drawdown = 0
        
        for i in range(len(drawdowns)):
            if not in_drawdown and drawdowns.iloc[i] < 0:
                in_drawdown = True
                start_idx = i
                max_drawdown = drawdowns.iloc[i]
            elif in_drawdown:
                max_drawdown = min(max_drawdown, drawdowns.iloc[i])
                if drawdowns.iloc[i] >= 0:
                    drawdown_periods.append({
                        'start_date': drawdowns.index[start_idx],
                        'end_date': drawdowns.index[i],
                        'duration_days': (drawdowns.index[i] - drawdowns.index[start_idx]).days,
                        'max_drawdown': max_drawdown,
                        'recovery_time': (drawdowns.index[i] - drawdowns.index[start_idx]).days
                    })
                    in_drawdown = False
                    
        return drawdown_periods
        
    def _calculate_benchmark_metrics(self,
                                   returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """
        Calcule les métriques relatives au benchmark.
        
        Args:
            returns: Série des rendements de la stratégie
            benchmark_returns: Série des rendements du benchmark
            
        Returns:
            Métriques relatives au benchmark
        """
        # Alignement des données
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return {}
            
        strategy_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        # Calcul du beta
        covariance = strategy_returns.cov(bench_returns)
        variance = bench_returns.var()
        beta = covariance / variance if variance != 0 else 0
        
        # Calcul de l'alpha
        risk_free_rate = 0.02  # Taux sans risque annuel
        daily_rf = risk_free_rate / 252
        alpha = (strategy_returns.mean() - daily_rf -
                beta * (bench_returns.mean() - daily_rf)) * 252
        
        # Calcul de l'information ratio
        tracking_error = (strategy_returns - bench_returns).std() * np.sqrt(252)
        information_ratio = ((strategy_returns - bench_returns).mean() * 252 /
                           tracking_error if tracking_error != 0 else 0)
        
        # Calcul des corrélations
        correlation = strategy_returns.corr(bench_returns)
        
        # Test de significativité
        t_stat, p_value = stats.ttest_ind(strategy_returns, bench_returns)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'relative_performance': (1 + strategy_returns).prod() / (1 + bench_returns).prod() - 1,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value
            }
        } 

def calculate_metrics(trades: List[Trade], 
                     equity_curve: pd.DataFrame,
                     initial_capital: float,
                     benchmark_returns: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calcule les métriques de performance à partir d'une liste de trades et d'une courbe d'équité.
    
    Args:
        trades: Liste des trades effectués
        equity_curve: Courbe d'équité
        initial_capital: Capital initial
        benchmark_returns: Rendements du benchmark (optionnel)
        risk_free_rate: Taux sans risque annualisé
        
    Returns:
        Dictionnaire contenant les métriques de performance
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_trades(
        trades=trades,
        equity_curve=equity_curve.to_dict('records'),
        initial_capital=initial_capital,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )