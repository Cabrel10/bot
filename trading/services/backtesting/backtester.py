from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.common.model_interface import ModelInterface
from ..data.data_types import ProcessedData, Position, Trade
from ..utils.logger import TradingLogger
from .performance_analyzer import PerformanceAnalyzer

class Backtester:
    """Backtesting de stratégies de trading."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le backtester.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.performance_analyzer = PerformanceAnalyzer()
        self._setup_logging()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration du backtester."""
        default_config = {
            'trading': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'slippage': 0.0001,
                'position_size': 0.1,  # Taille de position en % du capital
                'risk_management': {
                    'stop_loss': 0.02,  # Stop loss en %
                    'take_profit': 0.05,  # Take profit en %
                    'max_positions': 5,  # Nombre maximum de positions simultanées
                    'max_risk_per_trade': 0.01  # Risque maximum par trade en %
                }
            },
            'validation': {
                'min_trades': 20,
                'min_win_rate': 0.4,
                'max_drawdown': 0.2
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
        
    def _setup_logging(self) -> None:
        """Configure le système de logging."""
        logging.basicConfig(
            filename=f'logs/backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def run_backtest(self,
                    model: ModelInterface,
                    data: ProcessedData,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    initial_capital: Optional[float] = None) -> Dict:
        """
        Exécute un backtest sur une période donnée.
        
        Args:
            model: Modèle à tester
            data: Données de marché
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            initial_capital: Capital initial
            
        Returns:
            Résultats du backtest
        """
        try:
            # Préparation des données
            df = data.data.copy()
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # Initialisation du portefeuille
            initial_capital = initial_capital or self.config['trading']['initial_capital']
            portfolio = self._init_portfolio(initial_capital)
            
            # Simulation du trading
            trades = []
            positions = []
            equity_curve = []
            
            for timestamp, row in df.iterrows():
                # Mise à jour des positions existantes
                self._update_positions(positions, row, portfolio, trades)
                
                # Génération des signaux
                predictions = model.predict(row)
                
                # Ouverture de nouvelles positions
                if len(positions) < self.config['trading']['risk_management']['max_positions']:
                    new_positions = self._open_positions(predictions, row, portfolio)
                    positions.extend(new_positions)
                    
                # Enregistrement de l'équité
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': portfolio['cash'] + sum(pos.current_value for pos in positions)
                })
                
            # Analyse des performances
            results = self.performance_analyzer.analyze_trades(
                trades=trades,
                equity_curve=equity_curve,
                initial_capital=initial_capital
            )
            
            # Génération des visualisations
            results['charts'] = self._generate_charts(equity_curve, trades, df)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du backtest: {str(e)}")
            raise
            
    def _init_portfolio(self, initial_capital: float) -> Dict:
        """
        Initialise le portefeuille.
        
        Args:
            initial_capital: Capital initial
            
        Returns:
            État initial du portefeuille
        """
        return {
            'cash': initial_capital,
            'total_commission': 0.0,
            'total_slippage': 0.0
        }
        
    def _update_positions(self,
                        positions: List[Position],
                        current_data: pd.Series,
                        portfolio: Dict,
                        trades: List[Trade]) -> None:
        """
        Met à jour les positions existantes.
        
        Args:
            positions: Liste des positions ouvertes
            current_data: Données actuelles du marché
            portfolio: État du portefeuille
            trades: Liste des trades effectués
        """
        closed_positions = []
        
        for position in positions:
            # Mise à jour de la valeur
            position.current_value = position.quantity * current_data['close']
            position.unrealized_pnl = position.current_value - position.entry_value
            
            # Vérification des conditions de sortie
            if self._should_close_position(position, current_data):
                trade = self._close_position(position, current_data, portfolio)
                trades.append(trade)
                closed_positions.append(position)
                
        # Suppression des positions fermées
        for position in closed_positions:
            positions.remove(position)
            
    def _should_close_position(self, position: Position, current_data: pd.Series) -> bool:
        """
        Vérifie si une position doit être fermée.
        
        Args:
            position: Position à vérifier
            current_data: Données actuelles du marché
            
        Returns:
            True si la position doit être fermée
        """
        current_price = current_data['close']
        entry_price = position.entry_price
        
        # Stop loss
        stop_loss = self.config['trading']['risk_management']['stop_loss']
        if position.side == 'long':
            if current_price <= entry_price * (1 - stop_loss):
                return True
        else:
            if current_price >= entry_price * (1 + stop_loss):
                return True
                
        # Take profit
        take_profit = self.config['trading']['risk_management']['take_profit']
        if position.side == 'long':
            if current_price >= entry_price * (1 + take_profit):
                return True
        else:
            if current_price <= entry_price * (1 - take_profit):
                return True
                
        return False
        
    def _close_position(self,
                       position: Position,
                       current_data: pd.Series,
                       portfolio: Dict) -> Trade:
        """
        Ferme une position.
        
        Args:
            position: Position à fermer
            current_data: Données actuelles du marché
            portfolio: État du portefeuille
            
        Returns:
            Trade effectué
        """
        exit_price = current_data['close']
        
        # Calcul des coûts
        commission = abs(exit_price * position.quantity * self.config['trading']['commission_rate'])
        slippage = abs(exit_price * position.quantity * self.config['trading']['slippage'])
        
        # Mise à jour du portefeuille
        portfolio['cash'] += position.current_value - commission - slippage
        portfolio['total_commission'] += commission
        portfolio['total_slippage'] += slippage
        
        # Création du trade
        return Trade(
            entry_time=position.entry_time,
            exit_time=current_data.name,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=position.unrealized_pnl - commission - slippage,
            commission=commission,
            slippage=slippage
        )
        
    def _open_positions(self,
                       predictions: Dict,
                       current_data: pd.Series,
                       portfolio: Dict) -> List[Position]:
        """
        Ouvre de nouvelles positions.
        
        Args:
            predictions: Prédictions du modèle
            current_data: Données actuelles du marché
            portfolio: État du portefeuille
            
        Returns:
            Liste des nouvelles positions
        """
        new_positions = []
        
        for symbol, prediction in predictions.items():
            # Vérification du signal
            if abs(prediction['signal']) < prediction.get('threshold', 0.5):
                continue
                
            # Calcul de la taille de position
            position_value = portfolio['cash'] * self.config['trading']['position_size']
            if position_value <= 0:
                continue
                
            # Calcul des coûts
            quantity = position_value / current_data['close']
            commission = current_data['close'] * quantity * self.config['trading']['commission_rate']
            slippage = current_data['close'] * quantity * self.config['trading']['slippage']
            
            # Vérification du capital disponible
            if position_value + commission + slippage > portfolio['cash']:
                continue
                
            # Création de la position
            side = 'long' if prediction['signal'] > 0 else 'short'
            position = Position(
                entry_time=current_data.name,
                symbol=symbol,
                side=side,
                entry_price=current_data['close'],
                quantity=quantity,
                entry_value=position_value
            )
            
            # Mise à jour du portefeuille
            portfolio['cash'] -= position_value + commission + slippage
            portfolio['total_commission'] += commission
            portfolio['total_slippage'] += slippage
            
            new_positions.append(position)
            
        return new_positions
        
    def _generate_charts(self,
                        equity_curve: List[Dict],
                        trades: List[Trade],
                        market_data: pd.DataFrame) -> Dict:
        """
        Génère les visualisations du backtest.
        
        Args:
            equity_curve: Courbe d'équité
            trades: Liste des trades
            market_data: Données de marché
            
        Returns:
            Dictionnaire des figures Plotly
        """
        # Création des figures
        fig_equity = self._plot_equity_curve(equity_curve)
        fig_trades = self._plot_trades(trades, market_data)
        fig_metrics = self._plot_performance_metrics(trades)
        
        return {
            'equity_curve': fig_equity,
            'trades': fig_trades,
            'metrics': fig_metrics
        }
        
    def _plot_equity_curve(self, equity_curve: List[Dict]) -> go.Figure:
        """
        Génère le graphique de la courbe d'équité.
        
        Args:
            equity_curve: Données de la courbe d'équité
            
        Returns:
            Figure Plotly
        """
        df = pd.DataFrame(equity_curve)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Équité'
        ))
        
        fig.update_layout(
            title='Courbe d\'Équité',
            xaxis_title='Date',
            yaxis_title='Équité',
            showlegend=True
        )
        
        return fig
        
    def _plot_trades(self, trades: List[Trade], market_data: pd.DataFrame) -> go.Figure:
        """
        Génère le graphique des trades.
        
        Args:
            trades: Liste des trades
            market_data: Données de marché
            
        Returns:
            Figure Plotly
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True)
        
        # Prix et trades
        fig.add_trace(
            go.Candlestick(
                x=market_data.index,
                open=market_data['open'],
                high=market_data['high'],
                low=market_data['low'],
                close=market_data['close'],
                name='Prix'
            ),
            row=1, col=1
        )
        
        # Entrées et sorties
        for trade in trades:
            color = 'green' if trade.pnl > 0 else 'red'
            
            # Entrée
            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color=color),
                    name=f'Entrée ({trade.side})'
                ),
                row=1, col=1
            )
            
            # Sortie
            fig.add_trace(
                go.Scatter(
                    x=[trade.exit_time],
                    y=[trade.exit_price],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color=color),
                    name='Sortie'
                ),
                row=1, col=1
            )
            
        # PnL cumulé
        pnl_curve = pd.DataFrame([{
            'timestamp': trade.exit_time,
            'pnl': trade.pnl
        } for trade in trades])
        pnl_curve['cumulative_pnl'] = pnl_curve['pnl'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=pnl_curve['timestamp'],
                y=pnl_curve['cumulative_pnl'],
                mode='lines',
                name='PnL Cumulé'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Trades et PnL',
            xaxis_title='Date',
            yaxis_title='Prix',
            showlegend=True
        )
        
        return fig
        
    def _plot_performance_metrics(self, trades: List[Trade]) -> go.Figure:
        """
        Génère le graphique des métriques de performance.
        
        Args:
            trades: Liste des trades
            
        Returns:
            Figure Plotly
        """
        # Calcul des métriques mensuelles
        monthly_metrics = pd.DataFrame([{
            'timestamp': trade.exit_time,
            'pnl': trade.pnl
        } for trade in trades])
        monthly_metrics.set_index('timestamp', inplace=True)
        monthly_metrics = monthly_metrics.resample('M').agg({
            'pnl': ['sum', 'count', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
        })
        monthly_metrics.columns = ['pnl', 'trades', 'win_rate']
        
        # Création du graphique
        fig = make_subplots(rows=3, cols=1, shared_xaxis=True)
        
        # PnL mensuel
        fig.add_trace(
            go.Bar(
                x=monthly_metrics.index,
                y=monthly_metrics['pnl'],
                name='PnL Mensuel'
            ),
            row=1, col=1
        )
        
        # Nombre de trades
        fig.add_trace(
            go.Bar(
                x=monthly_metrics.index,
                y=monthly_metrics['trades'],
                name='Nombre de Trades'
            ),
            row=2, col=1
        )
        
        # Win rate
        fig.add_trace(
            go.Scatter(
                x=monthly_metrics.index,
                y=monthly_metrics['win_rate'],
                mode='lines+markers',
                name='Win Rate'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Métriques de Performance Mensuelles',
            showlegend=True
        )
        
        return fig
        
    def run_parallel_backtests(self,
                             models: List[ModelInterface],
                             data: ProcessedData,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Dict]:
        """
        Exécute des backtests en parallèle pour plusieurs modèles.
        
        Args:
            models: Liste des modèles à tester
            data: Données de marché
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            
        Returns:
            Résultats des backtests par modèle
        """
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_model = {
                executor.submit(
                    self.run_backtest,
                    model,
                    data,
                    start_date,
                    end_date
                ): model for model in models
            }
            
            for future in future_to_model:
                model = future_to_model[future]
                try:
                    results[model.__class__.__name__] = future.result()
                except Exception as e:
                    self.logger.error(f"Erreur pour {model.__class__.__name__}: {str(e)}")
                    
        return results
        
    def generate_backtest_report(self,
                               results: Dict[str, Dict],
                               output_path: Optional[str] = None) -> str:
        """
        Génère un rapport de backtest.
        
        Args:
            results: Résultats du backtest
            output_path: Chemin de sortie du rapport
            
        Returns:
            Chemin du rapport généré
        """
        if not output_path:
            output_path = f"reports/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
        try:
            # Création du rapport HTML
            with open(output_path, 'w') as f:
                f.write(self._generate_html_report(results))
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            raise
            
    def _generate_html_report(self, results: Dict[str, Dict]) -> str:
        """
        Génère le contenu HTML du rapport.
        
        Args:
            results: Résultats du backtest
            
        Returns:
            Contenu HTML
        """
        # Template HTML à implémenter
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Rapport de Backtest</title>
                <style>
                    /* Styles CSS à ajouter */
                </style>
            </head>
            <body>
                <h1>Rapport de Backtest</h1>
                <!-- Contenu du rapport à générer -->
            </body>
        </html>
        """ 