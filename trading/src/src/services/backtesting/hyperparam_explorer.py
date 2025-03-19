from itertools import product
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import pickle
import gzip
from datetime import datetime
import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from ..models.neural_network_model.model import NeuralTradingModel
from ..models.genetic_algorithm_model.model import GeneticTradingModel
from ..data.data_types import ProcessedData
from ..utils.logger import TradingLogger
from ..execution.risk_manager import RiskManager

class StrategyConfigGenerator:
    """Generates all parameter combinations for training"""
    
    def __init__(self, param_space: Dict[str, List]):
        self.param_space = param_space
        
    def generate_configs(self) -> List[Dict]:
        """Produces all possible parameter combinations"""
        keys = self.param_space.keys()
        values = product(*self.param_space.values())
        return [dict(zip(keys, v)) for v in values]

class TrainingResultAnalyzer:
    """Comparative analysis of training results"""
    
    def __init__(self, results: pd.DataFrame):
        self.results = results
        
    def get_top_strategies(self, 
                          metric: str = 'sharpe_ratio', 
                          n: int = 5) -> pd.DataFrame:
        """Returns the best strategies according to chosen metric"""
        return self.results.nlargest(n, metric)
    
    def analyze_sensitivity(self, param: str) -> pd.DataFrame:
        """Analyzes the sensitivity of performance to a specific parameter"""
        return self.results.groupby(param).mean()
    
    def get_correlation_matrix(self, metrics: List[str]) -> pd.DataFrame:
        """Generates correlation matrix between different performance metrics"""
        return self.results[metrics].corr()

class DataValidator:
    """Valide les données d'entrée pour l'optimisation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['data_validation']
    
    def validate(self, data: ProcessedData) -> Tuple[bool, List[str]]:
        """Valide les données selon les critères configurés."""
        errors = []
        
        # Vérification du nombre d'échantillons
        if len(data.data) < self.config['min_samples']:
            errors.append(f"Insufficient samples: {len(data.data)} < {self.config['min_samples']}")
        
        # Vérification des features requises
        missing_features = set(self.config['required_features']) - set(data.data.columns)
        if missing_features:
            errors.append(f"Missing features: {missing_features}")
        
        # Vérification de la couverture des données
        coverage = data.data.notna().mean()
        low_coverage = coverage[coverage < self.config['min_feature_coverage']]
        if not low_coverage.empty:
            errors.append(f"Low feature coverage: {low_coverage.to_dict()}")
        
        # Vérification des séries temporelles
        if self.config['time_series_checks']['check_gaps']:
            gaps = self._check_time_gaps(data)
            if gaps:
                errors.append(f"Time gaps found: {gaps}")
        
        return len(errors) == 0, errors
    
    def _check_time_gaps(self, data: ProcessedData) -> List[Dict[str, Any]]:
        """Vérifie les gaps dans les séries temporelles."""
        gaps = []
        timestamps = pd.to_datetime(data.data.index)
        diff = timestamps.diff()
        max_gap = pd.Timedelta(hours=self.config['time_series_checks']['max_gap_hours'])
        
        large_gaps = diff[diff > max_gap]
        if not large_gaps.empty:
            for idx, gap in large_gaps.items():
                gaps.append({
                    'start': timestamps[idx-1].isoformat(),
                    'end': timestamps[idx].isoformat(),
                    'duration_hours': gap.total_seconds() / 3600
                })
        
        return gaps

class ModelCheckpointing:
    """Gère la sauvegarde et la reprise des modèles."""
    
    def __init__(self, config: Dict[str, Any], base_dir: Path):
        self.config = config['optimization']['checkpointing']
        self.base_dir = base_dir / 'checkpoints'
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, study: Any, trial: Any) -> None:
        """Sauvegarde un point de contrôle."""
        if not self.config['enabled']:
            return
            
        if trial.number % self.config['frequency'] == 0:
            checkpoint_path = self.base_dir / f"checkpoint_{trial.number}.pkl.gz"
            
            with gzip.open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'study': study,
                    'trial_number': trial.number,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            self._cleanup_old_checkpoints()
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Charge le dernier point de contrôle."""
        if not self.config['enabled']:
            return None
            
        checkpoints = sorted(self.base_dir.glob("checkpoint_*.pkl.gz"))
        if not checkpoints:
            return None
            
        latest = checkpoints[-1]
        with gzip.open(latest, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Nettoie les anciens points de contrôle."""
        if self.config['cleanup_old']:
            checkpoints = sorted(self.base_dir.glob("checkpoint_*.pkl.gz"))
            if len(checkpoints) > self.config['max_checkpoints']:
                for checkpoint in checkpoints[:-self.config['max_checkpoints']]:
                    checkpoint.unlink()

class ModelArchive:
    """Gère l'archivage des modèles optimisés."""
    
    def __init__(self, config: Dict[str, Any], base_dir: Path):
        self.config = config['model_saving']
        self.base_dir = base_dir / 'models'
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, params: Dict[str, Any], 
                  metrics: Dict[str, float]) -> str:
        """Sauvegarde un modèle avec ses métadonnées."""
        if not self.config['save_best']:
            return ""
            
        # Création d'un identifiant unique
        model_id = self._generate_model_id(params)
        model_dir = self.base_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Sauvegarde du modèle
        model_path = model_dir / "model.pkl"
        if self.config['compression']:
            model_path = model_path.with_suffix('.pkl.gz')
            with gzip.open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Sauvegarde des métadonnées
        if self.config['include_metadata']:
            metadata = {
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'version': self.config.get('version', '1.0.0')
            }
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
        
        return model_id
    
    def _generate_model_id(self, params: Dict[str, Any]) -> str:
        """Génère un identifiant unique pour le modèle."""
        params_str = json.dumps(params, sort_keys=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_str = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{timestamp}_{hash_str}"

class HyperparamExplorer:
    """Explorateur d'hyperparamètres pour les modèles de trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise l'explorateur.
        
        Args:
            config_path: Chemin vers la configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.study = None
        self.best_params = None
        self.results_history: List[Dict[str, Any]] = []
        self.data_validator = DataValidator(self.config)
        self.checkpointing = ModelCheckpointing(self.config, Path(self.config['general']['results_dir']))
        self.model_archive = ModelArchive(self.config, Path(self.config['general']['results_dir']))
        self.risk_manager = RiskManager()

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Charge la configuration."""
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'hyperparam_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def optimize(self,
                      model_type: str,
                      train_data: ProcessedData,
                      val_data: ProcessedData,
                      n_trials: int = 100,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """Version améliorée avec validation et reprise."""
        try:
            # Validation des données
            is_valid, errors = self.data_validator.validate(train_data)
            if not is_valid:
                raise ValueError(f"Invalid training data: {errors}")
            
            is_valid, errors = self.data_validator.validate(val_data)
            if not is_valid:
                raise ValueError(f"Invalid validation data: {errors}")
            
            # Reprise possible d'une optimisation précédente
            checkpoint = self.checkpointing.load_latest_checkpoint()
            if checkpoint and self.config['general']['resume_enabled']:
                self.study = checkpoint['study']
                start_trial = checkpoint['trial_number'] + 1
            else:
                # Configuration de l'étude Optuna
                study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    sampler=TPESampler()
                )
                start_trial = 0
            
            # Sélection de la fonction objective selon le type de modèle
            if model_type == 'neural':
                objective = self._create_neural_objective(train_data, val_data)
            elif model_type == 'genetic':
                objective = self._create_genetic_objective(train_data, val_data)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")

            # Lancement de l'optimisation
            await self._run_optimization(objective, n_trials, timeout)

            # Sauvegarde et retour des résultats
            self.best_params = self.study.best_params
            self._save_results()
            
            # Sauvegarde du meilleur modèle
            if self.best_params:
                model_id = self.model_archive.save_model(
                    model=self.study.best_trial.user_attrs['model'],
                    params=self.best_params,
                    metrics={'value': self.study.best_value}
                )
            
            return {
                'best_params': self.best_params,
                'best_value': self.study.best_value,
                'n_trials': len(self.study.trials),
                'study_name': study_name,
                'model_id': model_id if self.best_params else None
            }

        except Exception as e:
            self.logger.log_error(e, {'action': 'optimize', 'model_type': model_type})
            raise

    def _create_neural_objective(self,
                               train_data: ProcessedData,
                               val_data: ProcessedData) -> Callable[[Trial], float]:
        """Crée la fonction objective pour le modèle neuronal."""
        
        async def objective(trial: Trial) -> float:
            try:
                # Définition de l'espace de recherche
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'batch_size': trial.suggest_int('batch_size', 16, 256, log=True),
                    'n_layers': trial.suggest_int('n_layers', 1, 4),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5)
                }
                
                # Paramètres des couches
                layers = []
                for i in range(params['n_layers']):
                    layers.append({
                        'units': trial.suggest_int(f'layer_{i}_units', 32, 256, log=True),
                        'activation': trial.suggest_categorical(
                            f'layer_{i}_activation',
                            ['relu', 'tanh', 'elu']
                        )
                    })
                params['layers'] = layers

                # Création et entraînement du modèle
                model = NeuralTradingModel(params)
                results = await model.train(train_data, val_data)
                
                # Sauvegarde des résultats
                self.results_history.append({
                    'params': params,
                    'metrics': results,
                    'timestamp': datetime.now().isoformat()
                })

                return results['val_score']

            except Exception as e:
                self.logger.log_error(e, {'action': 'neural_objective'})
                return float('-inf')

        return objective

    def _create_genetic_objective(self,
                                train_data: ProcessedData,
                                val_data: ProcessedData) -> Callable[[Trial], float]:
        """Crée la fonction objective pour le modèle génétique."""
        
        async def objective(trial: Trial) -> float:
            try:
                # Définition de l'espace de recherche
                params = {
                    'population_size': trial.suggest_int('population_size', 50, 500),
                    'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.3),
                    'crossover_rate': trial.suggest_float('crossover_rate', 0.6, 0.9),
                    'tournament_size': trial.suggest_int('tournament_size', 2, 8),
                    'elite_size': trial.suggest_int('elite_size', 1, 10)
                }

                # Création et entraînement du modèle
                model = GeneticTradingModel(params)
                results = await model.train(train_data, val_data)
                
                # Sauvegarde des résultats
                self.results_history.append({
                    'params': params,
                    'metrics': results,
                    'timestamp': datetime.now().isoformat()
                })

                return results['best_fitness']

            except Exception as e:
                self.logger.log_error(e, {'action': 'genetic_objective'})
                return float('-inf')

        return objective

    async def _run_optimization(self,
                              objective: Callable[[Trial], float],
                              n_trials: int,
                              timeout: Optional[int]) -> None:
        """Exécute l'optimisation avec Optuna."""
        try:
            # Création d'un wrapper pour l'objectif asynchrone
            def objective_wrapper(trial: Trial) -> float:
                return asyncio.run(objective(trial))

            # Exécution de l'optimisation
            self.study.optimize(
                objective_wrapper,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.config.get('n_jobs', 1)
            )

        except Exception as e:
            self.logger.log_error(e, {'action': 'run_optimization'})
            raise

    def _save_results(self) -> None:
        """Sauvegarde les résultats de l'optimisation."""
        try:
            results_dir = Path(self.config.get('results_dir', 'results/hyperopt'))
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde des meilleurs paramètres
            best_params_path = results_dir / f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(best_params_path, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_value': self.study.best_value,
                    'n_trials': len(self.study.trials),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=4)
            
            # Sauvegarde de l'historique complet
            history_path = results_dir / f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(history_path, 'w') as f:
                json.dump(self.results_history, f, indent=4)

        except Exception as e:
            self.logger.log_error(e, {'action': 'save_results'})

    def plot_optimization_history(self) -> None:
        """Affiche l'historique d'optimisation."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Création des figures
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=('Optimization History', 'Parameter Importance'))
            
            # Historique d'optimisation
            scores = [trial.value for trial in self.study.trials]
            best_scores = np.maximum.accumulate(scores)
            
            fig.add_trace(
                go.Scatter(y=scores, name='Trial Score',
                          mode='markers', marker=dict(size=6)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=best_scores, name='Best Score',
                          line=dict(color='red')),
                row=1, col=1
            )
            
            # Importance des paramètres
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())
            
            fig.add_trace(
                go.Bar(x=params, y=values, name='Parameter Importance'),
                row=2, col=1
            )
            
            # Mise à jour du layout
            fig.update_layout(height=800, title_text="Hyperparameter Optimization Results")
            fig.show()

        except Exception as e:
            self.logger.log_error(e, {'action': 'plot_optimization_history'})

    async def evaluate_model(self, model: BaseModel, train_data: MarketData, val_data: MarketData) -> Dict[str, Any]:
        """Évalue un modèle avec ses hyperparamètres."""
        try:
            # Entraînement du modèle
            train_results = await model.train(train_data, val_data)
            
            # Récupération des données de performance
            performance_data = await model.get_performance_data()
            
            # Évaluation des risques
            risk_assessment = await self.risk_manager.evaluate_model_risk(
                model=model,
                performance_data=performance_data
            )
            
            # Combinaison des résultats
            return {
                "training_results": train_results,
                "risk_assessment": risk_assessment,
                "performance_metrics": performance_data.metrics.__dict__
            }
            
        except Exception as e:
            self.logger.log_error(e, {'action': 'evaluate_model'})
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création de l'explorateur
        explorer = HyperparamExplorer()
        
        # Données d'exemple
        train_data = ProcessedData(...)  # À compléter
        val_data = ProcessedData(...)    # À compléter
        
        try:
            # Optimisation du modèle neuronal
            results = await explorer.optimize(
                model_type='neural',
                train_data=train_data,
                val_data=val_data,
                n_trials=50
            )
            
            print("Meilleurs paramètres trouvés:")
            print(json.dumps(results['best_params'], indent=2))
            
            # Visualisation des résultats
            explorer.plot_optimization_history()
            
        except Exception as e:
            print(f"Erreur: {e}")

    # Exécution
    asyncio.run(main())
