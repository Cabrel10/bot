import mlflow
import tkinter as tk
from tkinter import ttk
import json
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from ..strategies.trading_strategy import TradingStrategy
from ..visualization.dashboard import Dashboard
from ..data.historical_data import HistoricalDataManager
from ..data.data_types import TradeData, ModelPrediction, ProcessedData, PerformanceData
from ..models.common.base_model import BaseModel
from ..utils.logger import TradingLogger
from datetime import datetime
from ..execution.risk_manager import RiskManager

class StrategyApprovalSystem:
    def __init__(self, candidate_strategies: List[TradingStrategy]):
        self.candidate_strategies = candidate_strategies
        self.selections: List[TradingStrategy] = []
        self.dashboard = Dashboard()
        self._setup_mlflow()
        self.logger = TradingLogger()
        self.data_manager = HistoricalDataManager()
        self.results_history: List[Dict[str, Any]] = []
        self.risk_manager = RiskManager()

    def _setup_mlflow(self):
        """Configure MLflow for experiment tracking"""
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('strategy_validation')

    def show_interface(self):
        """Display the strategy approval interface"""
        self.root = tk.Tk()
        self.root.title('Strategy Approval System')
        self.root.geometry('800x600')

        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Strategy list
        self._create_strategy_list(main_container)

        # Performance metrics
        self._create_metrics_panel(main_container)

        # Action buttons
        self._create_action_buttons(main_container)

        self.root.mainloop()

    def _create_strategy_list(self, container):
        """Create the strategy selection list"""
        list_frame = ttk.LabelFrame(container, text='Available Strategies')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.strategy_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        self.strategy_listbox.pack(fill=tk.BOTH, expand=True)

        for strategy in self.candidate_strategies:
            self.strategy_listbox.insert(tk.END, strategy.name)

        self.strategy_listbox.bind('<<ListboxSelect>>', self._on_strategy_select)

    def _create_metrics_panel(self, container):
        """Create the performance metrics panel"""
        metrics_frame = ttk.LabelFrame(container, text='Performance Metrics')
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add metrics visualization from dashboard
        self.dashboard.create_metrics_view(metrics_frame)

    def _create_action_buttons(self, container):
        """Create action buttons"""
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text='Approve Selected', 
                   command=self._approve_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='View Details', 
                   command=self._view_details).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Close', 
                   command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

    def _on_strategy_select(self, event):
        """Handle strategy selection event"""
        selection = self.strategy_listbox.curselection()
        if selection:
            strategy = self.candidate_strategies[selection[0]]
            self._log_strategy_inspection(strategy)
            self.dashboard.update_metrics(strategy.get_performance_metrics())

    def _approve_selected(self):
        """Approve selected strategies"""
        selections = self.strategy_listbox.curselection()
        self.selections = [self.candidate_strategies[i] for i in selections]
        
        # Log approved strategies
        with mlflow.start_run(nested=True):
            for strategy in self.selections:
                mlflow.log_params({
                    f'strategy_{strategy.name}_config': json.dumps(strategy.get_config())
                })
                mlflow.log_metrics(strategy.get_performance_metrics())

        self.root.destroy()

    def _view_details(self):
        """Show detailed strategy information"""
        selection = self.strategy_listbox.curselection()
        if selection:
            strategy = self.candidate_strategies[selection[0]]
            self.dashboard.show_detailed_view(strategy)

    def _log_strategy_inspection(self, strategy: TradingStrategy):
        """Log strategy inspection event"""
        with mlflow.start_run(nested=True):
            mlflow.log_param('inspected_strategy', strategy.name)
            mlflow.log_param('inspection_timestamp', mlflow.utils.time.get_current_time_millis())

    async def _simulate_trades(self, 
                             predictions: List[ModelPrediction],
                             data: ProcessedData) -> List[TradeData]:
        """Simule les trades basés sur les prédictions.
        
        Args:
            predictions: Liste des prédictions du modèle
            data: Données de marché
            
        Returns:
            Liste des trades simulés
        """
        trades: List[TradeData] = []
        
        try:
            position_size = self.config['simulation']['position_size']
            stop_loss = self.config['simulation']['stop_loss']
            take_profit = self.config['simulation']['take_profit']
            
            current_position = None
            
            for i, pred in enumerate(predictions):
                current_price = data.data.iloc[i]['close']
                
                # Ouverture de position
                if not current_position and pred.signal != 0:
                    current_position = TradeData(
                        symbol=data.metadata['symbol'],
                        entry_time=data.data.index[i],
                        entry_price=current_price,
                        position_size=position_size * pred.signal,  # Long ou Short
                        stop_loss=current_price * (1 - stop_loss * pred.signal),
                        take_profit=current_price * (1 + take_profit * pred.signal)
                    )
                
                # Gestion de la position existante
                elif current_position:
                    # Vérification des conditions de sortie
                    if self._check_exit_conditions(
                        current_position, 
                        current_price,
                        data.data.index[i]
                    ):
                        trades.append(current_position)
                        current_position = None
            
            return trades
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'simulate_trades'})
            return []

    def _check_exit_conditions(self,
                             trade: TradeData,
                             current_price: float,
                             current_time: datetime) -> bool:
        """Vérifie les conditions de sortie d'une position.
        
        Args:
            trade: Trade en cours
            current_price: Prix actuel
            current_time: Temps actuel
            
        Returns:
            True si la position doit être fermée
        """
        try:
            # Stop Loss
            if trade.position_size > 0:  # Position longue
                if current_price <= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = current_time
                    trade.exit_reason = 'stop_loss'
                    return True
                elif current_price >= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_time = current_time
                    trade.exit_reason = 'take_profit'
                    return True
            else:  # Position courte
                if current_price >= trade.stop_loss:
                    trade.exit_price = trade.stop_loss
                    trade.exit_time = current_time
                    trade.exit_reason = 'stop_loss'
                    return True
                elif current_price <= trade.take_profit:
                    trade.exit_price = trade.take_profit
                    trade.exit_time = current_time
                    trade.exit_reason = 'take_profit'
                    return True
            
            return False
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'check_exit_conditions'})
            return True  # En cas d'erreur, on ferme la position

    def _calculate_trade_metrics(self, trades: List[TradeData]) -> Dict[str, float]:
        """Calcule les métriques de performance des trades.
        
        Args:
            trades: Liste des trades
            
        Returns:
            Dictionnaire des métriques
        """
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calcul des gains/pertes
            profits = []
            for trade in trades:
                if trade.exit_price and trade.entry_price:
                    profit = (trade.exit_price - trade.entry_price) * trade.position_size
                    profits.append(profit)
            
            # Métriques
            winning_trades = sum(1 for p in profits if p > 0)
            total_trades = len(profits)
            
            return {
                'total_trades': total_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'profit_factor': self._calculate_profit_factor(profits),
                'sharpe_ratio': self._calculate_sharpe_ratio(profits),
                'max_drawdown': self._calculate_max_drawdown(profits)
            }
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'calculate_trade_metrics'})
            return {}

    async def evaluate_strategy(self, strategy: TradingStrategy) -> Dict[str, Any]:
        """Évalue une stratégie candidate."""
        try:
            # Récupération des données de performance
            performance_data = await strategy.get_performance_data()
            
            # Évaluation des risques
            risk_assessment = await self.risk_manager.evaluate_model_risk(
                model=strategy,
                performance_data=performance_data
            )
            
            # Vérification des critères d'approbation
            approval_status = self._check_approval_criteria(
                risk_assessment,
                performance_data.metrics
            )
            
            return {
                "strategy_name": strategy.name,
                "risk_assessment": risk_assessment,
                "approval_status": approval_status,
                "recommendations": risk_assessment.get("recommendations", [])
            }
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'evaluate_strategy'})
            raise
