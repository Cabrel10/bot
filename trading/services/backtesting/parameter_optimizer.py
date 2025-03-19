from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import ParameterGrid
import yaml
import json

from ..models.neural_network_model.model import NeuralTradingModel
from ..models.genetic_algorithm_model.model import GeneticTradingModel
from ..data.data_types import ProcessedData
from ..utils.logger import TradingLogger

class ParameterOptimizer:
    """Optimiseur de paramètres pour les modèles de trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise l'optimiseur.
        
        Args:
            config_path: Chemin vers la configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Charge la configuration."""
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'parameter_optimizer_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def grid_search(self,
                         model_type: str,
                         param_grid: Dict[str, List[Any]],
                         train_data: ProcessedData,
                         val_data: ProcessedData,
                         metric: str = 'sharpe_ratio',
                         n_jobs: int = -1) -> Dict[str, Any]:
        """Effectue une recherche par grille des meilleurs paramètres.
        
        Args:
            model_type: Type de modèle ('neural' ou 'genetic')
            param_grid: Grille de paramètres à tester
            train_data: Données d'entraînement
            val_data: Données de validation
            metric: Métrique à optimiser
            n_jobs: Nombre de jobs parallèles (-1 pour tous les CPU)
            
        Returns:
            Résultats de l'optimisation
        """
        try:
            # Création de la grille de paramètres
            grid = list(ParameterGrid(param_grid))
            self.logger.log_info(f"Starting grid search with {len(grid)} combinations")

            # Parallélisation de l'évaluation
            with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
                tasks = [
                    self._evaluate_params(model_type, params, train_data, val_data, metric)
                    for params in grid
                ]
                results = await asyncio.gather(*tasks)

            # Analyse des résultats
            self._analyze_results(results)
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'all_results': self.results
            }

        except Exception as e:
            self.logger.log_error(e, {'action': 'grid_search'})
            raise

    async def random_search(self,
                          model_type: str,
                          param_distributions: Dict[str, Any],
                          n_iter: int,
                          train_data: ProcessedData,
                          val_data: ProcessedData,
                          metric: str = 'sharpe_ratio',
                          n_jobs: int = -1) -> Dict[str, Any]:
        """Effectue une recherche aléatoire des meilleurs paramètres.
        
        Args:
            model_type: Type de modèle
            param_distributions: Distributions des paramètres
            n_iter: Nombre d'itérations
            train_data: Données d'entraînement
            val_data: Données de validation
            metric: Métrique à optimiser
            n_jobs: Nombre de jobs parallèles
        """
        try:
            # Génération des combinaisons aléatoires
            param_combinations = [
                self._sample_params(param_distributions)
                for _ in range(n_iter)
            ]
            
            self.logger.log_info(f"Starting random search with {n_iter} iterations")

            # Parallélisation de l'évaluation
            with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
                tasks = [
                    self._evaluate_params(model_type, params, train_data, val_data, metric)
                    for params in param_combinations
                ]
                results = await asyncio.gather(*tasks)

            # Analyse des résultats
            self._analyze_results(results)
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'all_results': self.results
            }

        except Exception as e:
            self.logger.log_error(e, {'action': 'random_search'})
            raise

    def _sample_params(self, param_distributions: Dict[str, Any]) -> Dict[str, Any]:
        """Échantillonne des paramètres selon leurs distributions."""
        params = {}
        for param_name, dist in param_distributions.items():
            if isinstance(dist, dict):
                if dist.get('type') == 'float':
                    params[param_name] = np.random.uniform(
                        dist['min'], dist['max']
                    )
                elif dist.get('type') == 'int':
                    params[param_name] = np.random.randint(
                        dist['min'], dist['max'] + 1
                    )
                elif dist.get('type') == 'categorical':
                    params[param_name] = np.random.choice(dist['values'])
            elif isinstance(dist, list):
                params[param_name] = np.random.choice(dist)
            else:
                params[param_name] = dist
        
        return params

    async def _evaluate_params(self,
                             model_type: str,
                             params: Dict[str, Any],
                             train_data: ProcessedData,
                             val_data: ProcessedData,
                             metric: str) -> Dict[str, Any]:
        """Évalue une combinaison de paramètres."""
        try:
            # Création et entraînement du modèle
            if model_type == 'neural':
                model = NeuralTradingModel(params)
            elif model_type == 'genetic':
                model = GeneticTradingModel(params)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")

            # Entraînement
            train_results = await model.train(train_data, val_data)
            
            # Évaluation
            score = train_results.get(metric, float('-inf'))
            
            return {
                'params': params,
                'score': score,
                'metrics': train_results,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'evaluate_params',
                'params': params
            })
            return {
                'params': params,
                'score': float('-inf'),
                'error': str(e)
            }

    def _analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Analyse les résultats de l'optimisation."""
        try:
            # Mise à jour des résultats
            self.results.extend(results)
            
            # Tri des résultats par score
            valid_results = [r for r in results if 'error' not in r]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x['score'])
                
                if best_result['score'] > self.best_score:
                    self.best_score = best_result['score']
                    self.best_params = best_result['params']
            
            # Sauvegarde des résultats
            self._save_results()

        except Exception as e:
            self.logger.log_error(e, {'action': 'analyze_results'})

    def _save_results(self) -> None:
        """Sauvegarde les résultats de l'optimisation."""
        try:
            results_dir = Path(self.config['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Sauvegarde des résultats complets
            results_path = results_dir / f"optimization_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_score': self.best_score,
                    'all_results': self.results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=4)

        except Exception as e:
            self.logger.log_error(e, {'action': 'save_results'})

    def plot_results(self) -> None:
        """Visualise les résultats de l'optimisation."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Conversion en DataFrame
            df = pd.DataFrame([
                {**r['params'], 'score': r['score']}
                for r in self.results if 'error' not in r
            ])
            
            # Création des figures
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Parameter Distributions', 'Score Distribution')
            )
            
            # Distribution des paramètres
            for param in df.columns:
                if param != 'score':
                    fig.add_trace(
                        go.Box(y=df[param], name=param),
                        row=1, col=1
                    )
            
            # Distribution des scores
            fig.add_trace(
                go.Histogram(x=df['score'], name='Score Distribution'),
                row=2, col=1
            )
            
            # Mise à jour du layout
            fig.update_layout(height=800, title_text="Parameter Optimization Results")
            fig.show()

        except Exception as e:
            self.logger.log_error(e, {'action': 'plot_results'})

    def _validate_params(self, model_type: str, params: Dict[str, Any]) -> bool:
        """Valide les paramètres selon la configuration du modèle."""
        try:
            model_config_path = Path(__file__).parent.parent / 'models' / model_type / 'config.yaml'
            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
                
            param_constraints = model_config.get('param_constraints', {})
            
            for param_name, value in params.items():
                if param_name in param_constraints:
                    constraints = param_constraints[param_name]
                    
                    # Vérification des limites
                    if 'min' in constraints and value < constraints['min']:
                        return False
                    if 'max' in constraints and value > constraints['max']:
                        return False
                        
                    # Vérification du type
                    if 'type' in constraints:
                        expected_type = constraints['type']
                        if expected_type == 'int' and not isinstance(value, int):
                            return False
                        elif expected_type == 'float' and not isinstance(value, (int, float)):
                            return False
                            
                    # Vérification des valeurs autorisées
                    if 'allowed_values' in constraints and value not in constraints['allowed_values']:
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'validate_params'})
            return False

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création de l'optimiseur
        optimizer = ParameterOptimizer()
        
        # Exemple de grille de paramètres pour le modèle neuronal
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'n_layers': [1, 2],
            'dropout': [0.1, 0.3]
        }
        
        try:
            # Données d'exemple
            train_data = ProcessedData(...)  # À compléter
            val_data = ProcessedData(...)    # À compléter
            
            # Grid search
            results = await optimizer.grid_search(
                model_type='neural',
                param_grid=param_grid,
                train_data=train_data,
                val_data=val_data
            )
            
            print("Meilleurs paramètres trouvés:")
            print(json.dumps(results['best_params'], indent=2))
            
            # Visualisation
            optimizer.plot_results()
            
        except Exception as e:
            print(f"Erreur: {e}")

    # Exécution
    asyncio.run(main())