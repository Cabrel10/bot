from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from ..models.base_model import BaseModel
from ..core.data_types import MarketData
from ..utils.logger import TradingLogger

class ModelTrainingAnalyzer:
    """Analyseur spécialisé pour l'audit du processus d'entraînement."""
    
    def __init__(self):
        self.logger = TradingLogger()
        self._data_quality_metrics = {}
        self._training_metrics = {}
        
    def audit_data_pipeline(self, data: MarketData) -> Dict[str, Dict]:
        """Audit complet du pipeline de données.
        
        Args:
            data: Données à auditer
            
        Returns:
            Dict: Résultats de l'audit
        """
        try:
            results = {
                'data_quality': self._analyze_data_quality(data),
                'feature_analysis': self._analyze_features(data),
                'distribution_analysis': self._analyze_distributions(data),
                'temporal_analysis': self._analyze_temporal_aspects(data)
            }
            self._data_quality_metrics = results
            return results
        except Exception as e:
            self.logger.error(f"Erreur lors de l'audit des données: {str(e)}")
            raise
            
    def _analyze_data_quality(self, data: MarketData) -> Dict:
        """Analyse la qualité des données."""
        return {
            'missing_values': {
                'count': data.isnull().sum().to_dict(),
                'percentage': (data.isnull().sum() / len(data) * 100).to_dict()
            },
            'duplicates': {
                'count': len(data) - len(data.drop_duplicates()),
                'percentage': (1 - len(data.drop_duplicates()) / len(data)) * 100
            },
            'outliers': self._detect_outliers(data),
            'data_types': data.dtypes.to_dict(),
            'value_ranges': {
                col: {'min': data[col].min(), 'max': data[col].max()}
                for col in data.columns
            }
        }
        
    def _analyze_features(self, data: MarketData) -> Dict:
        """Analyse approfondie des features."""
        return {
            'correlations': data.corr().to_dict(),
            'feature_importance': self._calculate_feature_importance(data),
            'stationarity': self._test_stationarity(data),
            'collinearity': self._detect_collinearity(data)
        }
        
    def _analyze_distributions(self, data: MarketData) -> Dict:
        """Analyse des distributions."""
        return {
            'normality_tests': self._test_normality(data),
            'skewness': data.skew().to_dict(),
            'kurtosis': data.kurtosis().to_dict(),
            'distribution_params': self._fit_distributions(data)
        }
        
    def _analyze_temporal_aspects(self, data: MarketData) -> Dict:
        """Analyse des aspects temporels."""
        return {
            'seasonality': self._detect_seasonality(data),
            'trend': self._analyze_trends(data),
            'stationarity': self._check_stationarity(data),
            'autocorrelation': self._calculate_autocorrelation(data)
        }
        
    def audit_training_process(self, 
                             model: BaseModel,
                             training_data: MarketData,
                             validation_data: MarketData) -> Dict:
        """Audit du processus d'entraînement."""
        try:
            results = {
                'model_complexity': self._analyze_model_complexity(model),
                'training_stability': self._analyze_training_stability(model, training_data),
                'validation_metrics': self._analyze_validation_metrics(model, validation_data),
                'overfitting_analysis': self._analyze_overfitting(model, training_data, validation_data)
            }
            self._training_metrics = results
            return results
        except Exception as e:
            self.logger.error(f"Erreur lors de l'audit de l'entraînement: {str(e)}")
            raise
            
    def _analyze_model_complexity(self, model: BaseModel) -> Dict:
        """Analyse la complexité du modèle."""
        return {
            'parameter_count': model.get_parameter_count(),
            'layer_structure': model.get_layer_structure(),
            'activation_functions': model.get_activation_functions(),
            'regularization': model.get_regularization_params()
        }
        
    def _analyze_training_stability(self, 
                                  model: BaseModel,
                                  training_data: MarketData) -> Dict:
        """Analyse la stabilité de l'entraînement."""
        history = model.get_training_history()
        return {
            'loss_stability': self._analyze_loss_curve(history['loss']),
            'gradient_metrics': self._analyze_gradients(model),
            'learning_rate_analysis': self._analyze_learning_rate(history),
            'batch_size_impact': self._analyze_batch_size_impact(model, training_data)
        }
        
    def generate_training_report(self) -> str:
        """Génère un rapport détaillé de l'audit."""
        report = []
        
        # Analyse de la qualité des données
        report.append("## 1. Qualité des Données")
        report.append(self._format_data_quality_section())
        
        # Analyse des features
        report.append("## 2. Analyse des Features")
        report.append(self._format_feature_analysis())
        
        # Analyse de l'entraînement
        report.append("## 3. Processus d'Entraînement")
        report.append(self._format_training_analysis())
        
        # Recommandations
        report.append("## 4. Recommandations")
        report.append(self._generate_recommendations())
        
        return "\n\n".join(report)
        
    def visualize_training_metrics(self) -> go.Figure:
        """Visualise les métriques d'entraînement."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Qualité des Données',
                'Distribution des Features',
                'Courbes d\'Apprentissage',
                'Métriques de Validation'
            )
        )
        
        # Ajoute les visualisations
        self._add_data_quality_plot(fig, 1, 1)
        self._add_feature_distribution_plot(fig, 1, 2)
        self._add_learning_curves_plot(fig, 2, 1)
        self._add_validation_metrics_plot(fig, 2, 2)
        
        fig.update_layout(height=800, showlegend=True)
        return fig
        
    def _detect_outliers(self, data: MarketData) -> Dict:
        """Détecte les outliers dans les données."""
        outliers = {}
        for column in data.columns:
            if np.issubdtype(data[column].dtype, np.number):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers[column] = {
                    'count': len(data[(data[column] < Q1 - 1.5 * IQR) | 
                                    (data[column] > Q3 + 1.5 * IQR)]),
                    'percentage': len(data[(data[column] < Q1 - 1.5 * IQR) | 
                                        (data[column] > Q3 + 1.5 * IQR)]) / len(data) * 100
                }
        return outliers
        
    def _calculate_feature_importance(self, data: MarketData) -> Dict:
        """Calcule l'importance des features."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        return dict(zip(X.columns, rf.feature_importances_))
        
    def _test_stationarity(self, data: MarketData) -> Dict:
        """Teste la stationnarité des séries."""
        from statsmodels.tsa.stattools import adfuller
        
        stationarity = {}
        for column in data.columns:
            if np.issubdtype(data[column].dtype, np.number):
                result = adfuller(data[column].dropna())
                stationarity[column] = {
                    'test_statistic': result[0],
                    'p_value': result[1],
                    'is_stationary': result[1] < 0.05
                }
        return stationarity