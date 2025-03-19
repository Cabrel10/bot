from typing import Dict, List, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import ParameterGrid
from bayes_opt import BayesianOptimization
import optuna

from ..models.common.model_interface import ModelInterface
from ..data.data_types import ProcessedData
from ..utils.logger import TradingLogger
from .backtester import Backtester

class HyperparamTuner:
    """Optimisation des hyperparamètres des modèles de trading."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le tuner d'hyperparamètres.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.backtester = Backtester()
        self._setup_logging()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Charge la configuration du tuner."""
        default_config = {
            'optimization': {
                'method': 'bayesian',  # 'grid', 'random', 'bayesian', 'optuna'
                'n_trials': 100,
                'n_jobs': 4,
                'timeout': 3600,
                'cross_validation': {
                    'n_splits': 5,
                    'shuffle': True
                }
            },
            'evaluation': {
                'metric': 'sharpe_ratio',  # Métrique à optimiser
                'constraints': {
                    'min_trades': 20,
                    'max_drawdown': 0.2
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
        
    def _setup_logging(self) -> None:
        """Configure le système de logging."""
        logging.basicConfig(
            filename=f'logs/tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def optimize(self,
                model: ModelInterface,
                data: ProcessedData,
                param_space: Dict,
                objective_func: Optional[Callable] = None) -> Dict:
        """
        Optimise les hyperparamètres d'un modèle.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            param_space: Espace des hyperparamètres
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        method = self.config['optimization']['method']
        
        if method == 'grid':
            return self._grid_search(model, data, param_space, objective_func)
        elif method == 'random':
            return self._random_search(model, data, param_space, objective_func)
        elif method == 'bayesian':
            return self._bayesian_optimization(model, data, param_space, objective_func)
        elif method == 'optuna':
            return self._optuna_optimization(model, data, param_space, objective_func)
        else:
            raise ValueError(f"Méthode d'optimisation non supportée: {method}")
            
    def _grid_search(self,
                    model: ModelInterface,
                    data: ProcessedData,
                    param_space: Dict,
                    objective_func: Optional[Callable]) -> Dict:
        """
        Effectue une recherche par grille.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            param_space: Espace des hyperparamètres
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        best_score = float('-inf')
        best_params = None
        
        param_grid = ParameterGrid(param_space)
        n_jobs = self.config['optimization']['n_jobs']
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_params = {
                executor.submit(self._evaluate_params, model, data, params, objective_func): params
                for params in param_grid
            }
            
            for future in future_to_params:
                params = future_to_params[future]
                try:
                    score = future.result()
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                    self.logger.info(f"Params: {params}, Score: {score:.4f}")
                except Exception as e:
                    self.logger.error(f"Erreur pour {params}: {str(e)}")
                    
        return best_params
        
    def _random_search(self,
                      model: ModelInterface,
                      data: ProcessedData,
                      param_space: Dict,
                      objective_func: Optional[Callable]) -> Dict:
        """
        Effectue une recherche aléatoire.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            param_space: Espace des hyperparamètres
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        best_score = float('-inf')
        best_params = None
        n_trials = self.config['optimization']['n_trials']
        
        for _ in range(n_trials):
            params = self._sample_params(param_space)
            score = self._evaluate_params(model, data, params, objective_func)
            
            if score > best_score:
                best_score = score
                best_params = params
                
            self.logger.info(f"Params: {params}, Score: {score:.4f}")
            
        return best_params
        
    def _bayesian_optimization(self,
                             model: ModelInterface,
                             data: ProcessedData,
                             param_space: Dict,
                             objective_func: Optional[Callable]) -> Dict:
        """
        Effectue une optimisation bayésienne.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            param_space: Espace des hyperparamètres
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        def objective(**params):
            return self._evaluate_params(model, data, params, objective_func)
            
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_space,
            random_state=42
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=self.config['optimization']['n_trials']
        )
        
        return optimizer.max['params']
        
    def _optuna_optimization(self,
                           model: ModelInterface,
                           data: ProcessedData,
                           param_space: Dict,
                           objective_func: Optional[Callable]) -> Dict:
        """
        Effectue une optimisation avec Optuna.
        
        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            param_space: Espace des hyperparamètres
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        def objective(trial):
            params = {}
            for name, space in param_space.items():
                if isinstance(space, list):
                    if isinstance(space[0], int):
                        params[name] = trial.suggest_int(name, space[0], space[1])
                    else:
                        params[name] = trial.suggest_float(name, space[0], space[1])
                elif isinstance(space, dict):
                    if space['type'] == 'categorical':
                        params[name] = trial.suggest_categorical(name, space['values'])
                        
            return self._evaluate_params(model, data, params, objective_func)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization']['timeout']
        )
        
        return study.best_params
        
    def _evaluate_params(self,
                        model: ModelInterface,
                        data: ProcessedData,
                        params: Dict,
                        objective_func: Optional[Callable]) -> float:
        """
        Évalue un ensemble de paramètres.
        
        Args:
            model: Modèle à évaluer
            data: Données d'entraînement
            params: Paramètres à évaluer
            objective_func: Fonction objectif personnalisée
            
        Returns:
            Score d'évaluation
        """
        try:
            # Configuration du modèle
            model_copy = model.__class__()  # Crée une nouvelle instance
            model_copy.set_parameters(params)
            
            # Entraînement et évaluation
            if objective_func:
                return objective_func(model_copy, data)
                
            # Évaluation par défaut avec backtesting
            results = self.backtester.run_backtest(
                model=model_copy,
                data=data,
                start_date=data.data.index[0],
                end_date=data.data.index[-1]
            )
            
            # Vérification des contraintes
            if not self._check_constraints(results):
                return float('-inf')
                
            # Retourne la métrique spécifiée
            metric = self.config['evaluation']['metric']
            return results['performance'][metric]
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation: {str(e)}")
            return float('-inf')
            
    def _check_constraints(self, results: Dict) -> bool:
        """
        Vérifie si les résultats respectent les contraintes.
        
        Args:
            results: Résultats du backtest
            
        Returns:
            True si les contraintes sont respectées
        """
        constraints = self.config['evaluation']['constraints']
        
        if results['performance']['num_trades'] < constraints['min_trades']:
            return False
            
        if abs(results['performance']['max_drawdown']) > constraints['max_drawdown']:
            return False
            
        return True
        
    def _sample_params(self, param_space: Dict) -> Dict:
        """
        Échantillonne aléatoirement des paramètres.
        
        Args:
            param_space: Espace des paramètres
            
        Returns:
            Paramètres échantillonnés
        """
        params = {}
        for name, space in param_space.items():
            if isinstance(space, list):
                if isinstance(space[0], int):
                    params[name] = np.random.randint(space[0], space[1] + 1)
                else:
                    params[name] = np.random.uniform(space[0], space[1])
            elif isinstance(space, dict) and space['type'] == 'categorical':
                params[name] = np.random.choice(space['values'])
                
        return params
        
    def generate_tuning_report(self,
                             results: Dict[str, Dict],
                             param_space: Dict) -> str:
        """
        Génère un rapport d'optimisation.
        
        Args:
            results: Résultats de l'optimisation
            param_space: Espace des paramètres
            
        Returns:
            Chemin du rapport généré
        """
        try:
            report_path = f"reports/tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            # Création du rapport HTML
            with open(report_path, 'w') as f:
                f.write(self._generate_html_report(results, param_space))
                
            return report_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            raise
            
    def _generate_html_report(self, results: Dict[str, Dict], param_space: Dict) -> str:
        """
        Génère le contenu HTML du rapport.
        
        Args:
            results: Résultats de l'optimisation
            param_space: Espace des paramètres
            
        Returns:
            Contenu HTML
        """
        # Template HTML à implémenter
        return """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Rapport d'Optimisation</title>
                <style>
                    /* Styles CSS à ajouter */
                </style>
            </head>
            <body>
                <h1>Rapport d'Optimisation des Hyperparamètres</h1>
                <!-- Contenu du rapport à générer -->
            </body>
        </html>
        """ 