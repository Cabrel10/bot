"""
Composant de visualisation des résultats de backtest.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List

class BacktestVisualization:
    """Visualisation des résultats de backtest."""
    
    def __init__(self):
        """Initialisation du composant."""
        self.colors = {
            "profit": "rgba(0, 255, 0, 0.7)",
            "loss": "rgba(255, 0, 0, 0.7)",
            "equity": "rgba(0, 100, 255, 0.8)",
            "drawdown": "rgba(255, 0, 0, 0.4)"
        }
    
    def display_results(self, results):
        """Affiche les résultats du backtest."""
        st.header("Résultats du Backtest")
        
        # 1. Métriques principales
        self._display_metrics(results.metrics)
        
        # 2. Graphiques
        self._plot_equity_curve(results.equity_curve)
        self._plot_trades_distribution(results.trades)
        
        # 3. Performance par symbole
        self._display_symbol_performance(results.symbol_performance)
        
        # 4. Détails des trades
        self._display_trades_details(results.trades)
    
    def _display_metrics(self, metrics: Dict[str, float]):
        """Affiche les métriques principales."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Total", f"{metrics['total_return']:.2f}%")
            
        with col2:
            win_rate = (metrics['winning_trades'] / metrics['total_trades']) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
            
        with col3:
            st.metric("Ratio de Sharpe", f"{metrics['sharpe_ratio']:.2f}")
            
        with col4:
            st.metric("Drawdown Max", f"{metrics['max_drawdown']:.2f}%")
            
        # Métriques additionnelles
        with st.expander("Métriques détaillées"):
            st.write({
                "Nombre total de trades": metrics['total_trades'],
                "Trades gagnants": metrics['winning_trades'],
                "Profit Factor": f"{metrics['profit_factor']:.2f}"
            })
    
    def _plot_equity_curve(self, equity_df: pd.DataFrame):
        """Trace la courbe d'equity et le drawdown."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=("Courbe d'Equity", "Drawdown"))
        
        # Courbe d'equity
        fig.add_trace(
            go.Scatter(
                x=equity_df["timestamp"],
                y=equity_df["equity"],
                name="Equity",
                line=dict(color=self.colors["equity"])
            ),
            row=1, col=1
        )
        
        # Calcul et tracé du drawdown
        peak = equity_df["equity"].expanding(min_periods=1).max()
        drawdown = ((equity_df["equity"] - peak) / peak) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_df["timestamp"],
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color=self.colors["drawdown"])
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Performance du Trading"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_trades_distribution(self, trades_df: pd.DataFrame):
        """Trace la distribution des trades."""
        if trades_df.empty:
            return
            
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=("Distribution des Profits",
                                         "Profits par Symbole"))
        
        # Distribution des profits
        profits = trades_df["pnl"]
        fig.add_trace(
            go.Histogram(
                x=profits,
                name="Distribution",
                marker_color=self.colors["equity"]
            ),
            row=1, col=1
        )
        
        # Profits par symbole
        symbol_profits = trades_df.groupby("symbol")["pnl"].sum()
        colors = [self.colors["profit"] if x > 0 else self.colors["loss"]
                 for x in symbol_profits]
        
        fig.add_trace(
            go.Bar(
                x=symbol_profits.index,
                y=symbol_profits.values,
                name="Profit par Symbole",
                marker_color=colors
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Analyse des Trades"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_symbol_performance(self, performance: Dict[str, Dict[str, float]]):
        """Affiche les performances par symbole."""
        st.subheader("Performance par Symbole")
        
        # Création du DataFrame
        df = pd.DataFrame.from_dict(performance, orient="index")
        
        # Calcul du win rate
        df["win_rate"] = (df["winning_trades"] / df["total_trades"]) * 100
        
        # Formatage
        df = df.round(2)
        df["win_rate"] = df["win_rate"].map("{:.1f}%".format)
        df["avg_return"] = df["avg_return"].map("{:.2f}%".format)
        
        # Affichage
        st.dataframe(df)
    
    def _display_trades_details(self, trades_df: pd.DataFrame):
        """Affiche les détails des trades."""
        st.subheader("Détails des Trades")
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            symbols = st.multiselect(
                "Symboles",
                options=trades_df["symbol"].unique(),
                default=trades_df["symbol"].unique()
            )
        
        with col2:
            trade_type = st.selectbox(
                "Type de Trade",
                options=["Tous", "Gagnants", "Perdants"]
            )
        
        # Filtrage
        filtered_df = trades_df[trades_df["symbol"].isin(symbols)]
        if trade_type == "Gagnants":
            filtered_df = filtered_df[filtered_df["pnl"] > 0]
        elif trade_type == "Perdants":
            filtered_df = filtered_df[filtered_df["pnl"] < 0]
        
        # Formatage
        display_df = filtered_df.copy()
        display_df["return"] = display_df["return"].map("{:.2f}%".format)
        display_df["pnl"] = display_df["pnl"].map("{:.2f}".format)
        
        # Affichage
        st.dataframe(
            display_df.sort_values("entry_time", ascending=False),
            use_container_width=True
        ) 