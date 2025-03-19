import numpy as np
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from SALib.sample import saltelli
from SALib.analyze import sobol
from typing import Dict, List, Tuple, Callable, Optional
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SystematicOptimizer:
    """Optimisation systématique des modèles de trading."""
    
    def __init__(self, save_dir: str = "logs/optimization"):
        """
        Initialise l'optimiseur systématique.
        
        Args:
            save_dir: Répertoire de sauvegarde des résultats
        """
        self.save_dir = save_dir
        self._setup_logging()
        self.results_history = []
    
    def _setup_logging(self) -> None:
        """Configure le système de logging."""
        logging.basicConfig(
            filename=f'{self.save_dir}/optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def grid_search(self, param_grid: Dict, 
                   objective_func: Callable,
                   n_trials: int = 1) -> Dict:
        """
        Effectue une recherche par grille sur les paramètres.
        
        Args:
            param_grid: Dictionnaire des paramètres à tester
            objective_func: Fonction objectif à optimiser
            n_trials: Nombre d'essais par combinaison
            
        Returns:
            Meilleurs paramètres trouvés
        """
        best_score = float('-inf')
        best_params = None
        results = []
        
        for params in ParameterGrid(param_grid):
            scores = []
            for _ in range(n_trials):
                score = objective_func(params)
                scores.append(score)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            results.append({
                'params': params,
                'score_mean': avg_score,
                'score_std': std_score
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
            
            logging.info(f"Params: {params}, Score: {avg_score:.4f} ± {std_score:.4f}")
        
        self.results_history.extend(results)
        self._plot_grid_results(results)
        return best_params
    
    def bayesian_optimization(self, param_bounds: List[Tuple[float, float]],
                            param_names: List[str],
                            objective_func: Callable,
                            n_calls: int = 50) -> Dict:
        """
        Effectue une optimisation bayésienne.
        
        Args:
            param_bounds: Liste des bornes pour chaque paramètre
            param_names: Noms des paramètres
            objective_func: Fonction objectif à optimiser
            n_calls: Nombre d'appels à la fonction objectif
            
        Returns:
            Meilleurs paramètres trouvés
        """
        def objective_wrapper(params):
            param_dict = dict(zip(param_names, params))
            return -objective_func(param_dict)  # Négation car gp_minimize minimise
        
        result = gp_minimize(
            objective_wrapper,
            param_bounds,
            n_calls=n_calls,
            noise=0.1,
            n_random_starts=10
        )
        
        best_params = dict(zip(param_names, result.x))
        
        # Sauvegarde des résultats
        all_results = []
        for i, params in enumerate(result.x_iters):
            all_results.append({
                'params': dict(zip(param_names, params)),
                'score': -result.func_vals[i]
            })
        
        self.results_history.extend(all_results)
        self._plot_optimization_progress(result)
        
        return best_params
    
    def sensitivity_analysis(self, problem_def: Dict,
                           objective_func: Callable,
                           n_samples: int = 1000) -> Dict:
        """
        Effectue une analyse de sensibilité.
        
        Args:
            problem_def: Définition du problème pour SALib
            objective_func: Fonction objectif à évaluer
            n_samples: Nombre d'échantillons
            
        Returns:
            Résultats de l'analyse de sensibilité
        """
        # Génération des échantillons
        param_values = saltelli.sample(problem_def, n_samples)
        
        # Évaluation
        Y = np.array([objective_func(dict(zip(problem_def['names'], params))) 
                     for params in param_values])
        
        # Analyse
        sensitivity = sobol.analyze(problem_def, Y)
        
        # Formatage des résultats
        results = {}
        for i, param in enumerate(problem_def['names']):
            results[param] = {
                'S1': sensitivity['S1'][i],
                'S1_conf': sensitivity['S1_conf'][i],
                'ST': sensitivity['ST'][i],
                'ST_conf': sensitivity['ST_conf'][i]
            }
        
        self._plot_sensitivity_results(results)
        return results
    
    def _plot_grid_results(self, results: List[Dict]) -> None:
        """
        Visualise les résultats de la recherche par grille.
        
        Args:
            results: Liste des résultats
        """
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='params', y='score_mean')
        plt.xticks(rotation=45)
        plt.title('Distribution des Scores par Configuration')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/grid_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def _plot_optimization_progress(self, result) -> None:
        """
        Visualise la progression de l'optimisation bayésienne.
        
        Args:
            result: Résultat de l'optimisation
        """
        plt.figure(figsize=(10, 6))
        plt.plot(-result.func_vals)
        plt.title('Progression de l\'Optimisation Bayésienne')
        plt.xlabel('Itération')
        plt.ylabel('Score')
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/bayesian_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    
    def _plot_sensitivity_results(self, results: Dict) -> None:
        """
        Visualise les résultats de l'analyse de sensibilité.
        
        Args:
            results: Résultats de l'analyse
        """
        params = list(results.keys())
        s1_values = [results[p]['S1'] for p in params]
        st_values = [results[p]['ST'] for p in params]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(params))
        width = 0.35
        
        plt.bar(x - width/2, s1_values, width, label='Premier Ordre (S1)')
        plt.bar(x + width/2, st_values, width, label='Effet Total (ST)')
        
        plt.xlabel('Paramètres')
        plt.ylabel('Indices de Sensibilité')
        plt.title('Analyse de Sensibilité')
        plt.xticks(x, params, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/sensitivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()