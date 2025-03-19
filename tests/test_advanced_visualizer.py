import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from trading.visualization.advanced_visualizer import AdvancedVisualizer
from trading.models.base_model import BaseModel
from trading.core.data_types import MarketData, Trade

class MockModel(BaseModel):
    """Modèle fictif pour les tests."""
    def get_all_trades(self):
        return [
            Trade(
                symbol="BTC/USD",
                entry_price=50000,
                exit_price=51000,
                entry_time=datetime.now() - timedelta(days=1),
                exit_time=datetime.now(),
                side="BUY",
                quantity=1.0,
                pnl=1000
            )
        ]

class TestAdvancedVisualizer(unittest.TestCase):
    """Tests pour le visualiseur avancé."""
    
    def setUp(self):
        """Initialise l'environnement de test."""
        # Création des données de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        self.test_data = pd.DataFrame({
            'close': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100),
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'equity': np.linspace(100, 150, 100)
        }, index=dates)
        
        self.market_data = MarketData(self.test_data)
        self.model = MockModel()
        self.visualizer = AdvancedVisualizer(
            model=self.model,
            data=self.market_data,
            save_dir="tests/temp_viz"
        )
        
    def test_plot_shap_values(self):
        """Teste la génération du graphique SHAP."""
        fig = self.visualizer.plot_shap_values(
            feature_names=['feature1', 'feature2']
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) > 0)
        
    def test_plot_live_performance(self):
        """Teste la génération du graphique de performance."""
        fig = self.visualizer.plot_live_performance(
            interval=60,
            window=50
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) > 0)
        
    def test_plot_correlation_heatmap(self):
        """Teste la génération de la heatmap des corrélations."""
        fig = self.visualizer.plot_correlation_heatmap()
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) > 0)
        
    def test_plot_trade_analysis(self):
        """Teste la génération de l'analyse des trades."""
        trades = self.model.get_all_trades()
        fig = self.visualizer.plot_trade_analysis(
            trades=trades,
            market_data=self.market_data
        )
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(len(fig.data) > 0)
        
    def test_create_dash_components(self):
        """Teste la création des composants Dash."""
        components = self.visualizer.create_dash_components()
        self.assertIsNotNone(components)
        self.assertTrue(hasattr(components, 'children'))
        
    def test_calculate_drawdown(self):
        """Teste le calcul du drawdown."""
        equity = pd.Series(self.test_data['equity'])
        drawdown = self.visualizer._calculate_drawdown(equity)
        self.assertIsInstance(drawdown, pd.Series)
        self.assertTrue(all(drawdown <= 0))
        
    def test_error_handling(self):
        """Teste la gestion des erreurs."""
        # Test avec données invalides
        with self.assertRaises(ValueError):
            invalid_data = pd.DataFrame()
            self.visualizer.plot_correlation_heatmap()
            
    def test_cache_behavior(self):
        """Teste le comportement du cache."""
        # Premier appel
        fig1 = self.visualizer.plot_shap_values()
        # Deuxième appel (devrait utiliser le cache)
        fig2 = self.visualizer.plot_shap_values()
        self.assertEqual(id(fig1), id(fig2))
        
    def test_performance_metrics(self):
        """Teste le calcul des métriques de performance."""
        trades = self.model.get_all_trades()
        metrics = self.visualizer._calculate_performance_metrics(trades)
        
        required_metrics = [
            'total_return',
            'win_rate',
            'sharpe_ratio',
            'max_drawdown'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            
    def tearDown(self):
        """Nettoie l'environnement après les tests."""
        import shutil
        try:
            shutil.rmtree("tests/temp_viz")
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main() 