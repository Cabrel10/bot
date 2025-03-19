"""Tests unitaires pour le module multi_objective_optimizer."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Tuple

from src.core.models.multi_objective_optimizer import (
    FuturesConfig,
    VolumeConfig,
    TimeSeriesValidationConfig,
    ObjectiveConfig,
    NSGAConfig,
    Solution,
    NSGAII,
    ValidationError,
    OptimizationError,
    ConfigurationError
)

class TestFuturesConfig(unittest.TestCase):
    """Tests pour la classe FuturesConfig."""
    
    def test_valid_config(self):
        """Teste une configuration valide."""
        config = FuturesConfig(
            enabled=True,
            margin_requirement=0.1,
            contract_size=100.0,
            commission_rate=0.0001,
            slippage=0.0001
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.margin_requirement, 0.1)
        self.assertEqual(config.contract_size, 100.0)
        
    def test_invalid_margin_requirement(self):
        """Teste la validation de l'exigence de marge."""
        with self.assertRaises(ValidationError):
            FuturesConfig(enabled=True, margin_requirement=1.5)
            
    def test_invalid_contract_size(self):
        """Teste la validation de la taille du contrat."""
        with self.assertRaises(ValidationError):
            FuturesConfig(enabled=True, contract_size=0)
            
    def test_invalid_commission_rate(self):
        """Teste la validation du taux de commission."""
        with self.assertRaises(ValidationError):
            FuturesConfig(enabled=True, commission_rate=-0.1)
            
    def test_disabled_config(self):
        """Teste que la validation est ignorée si disabled."""
        config = FuturesConfig(enabled=False, margin_requirement=1.5)
        self.assertFalse(config.enabled)

class TestVolumeConfig(unittest.TestCase):
    """Tests pour la classe VolumeConfig."""
    
    def test_valid_config(self):
        """Teste une configuration valide."""
        config = VolumeConfig(
            enabled=True,
            min_volume=1000.0,
            volume_ma_periods=[20, 50, 200],
            volume_impact=0.3
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.min_volume, 1000.0)
        self.assertEqual(config.volume_ma_periods, [20, 50, 200])
        
    def test_invalid_min_volume(self):
        """Teste la validation du volume minimum."""
        with self.assertRaises(ValidationError):
            VolumeConfig(enabled=True, min_volume=0)
            
    def test_invalid_ma_periods(self):
        """Teste la validation des périodes MA."""
        with self.assertRaises(ValidationError):
            VolumeConfig(enabled=True, volume_ma_periods=[])
            
        with self.assertRaises(ValidationError):
            VolumeConfig(enabled=True, volume_ma_periods=[0, 20])
            
    def test_invalid_volume_impact(self):
        """Teste la validation de l'impact du volume."""
        with self.assertRaises(ValidationError):
            VolumeConfig(enabled=True, volume_impact=1.5)

class TestTimeSeriesValidationConfig(unittest.TestCase):
    """Tests pour la classe TimeSeriesValidationConfig."""
    
    def test_valid_config(self):
        """Teste une configuration valide."""
        config = TimeSeriesValidationConfig(
            enabled=True,
            n_splits=5,
            train_size=0.8,
            gap=1
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.n_splits, 5)
        self.assertEqual(config.train_size, 0.8)
        
    def test_invalid_n_splits(self):
        """Teste la validation du nombre de splits."""
        with self.assertRaises(ValidationError):
            TimeSeriesValidationConfig(enabled=True, n_splits=1)
            
    def test_invalid_train_size(self):
        """Teste la validation de la taille d'entraînement."""
        with self.assertRaises(ValidationError):
            TimeSeriesValidationConfig(enabled=True, train_size=1.5)
            
    def test_invalid_gap(self):
        """Teste la validation du gap."""
        with self.assertRaises(ValidationError):
            TimeSeriesValidationConfig(enabled=True, gap=-1)

class TestNSGAConfig(unittest.TestCase):
    """Tests pour la classe NSGAConfig."""
    
    def setUp(self):
        """Initialise les configurations pour les tests."""
        self.objectives = [
            ObjectiveConfig("return", weight=0.6),
            ObjectiveConfig("risk", weight=0.4, minimize=True)
        ]
        
    def test_valid_config(self):
        """Teste une configuration valide."""
        config = NSGAConfig(
            population_size=100,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            tournament_size=3,
            objectives=self.objectives
        )
        self.assertEqual(config.population_size, 100)
        self.assertEqual(len(config.objectives), 2)
        
    def test_invalid_population_size(self):
        """Teste la validation de la taille de population."""
        with self.assertRaises(ValidationError):
            NSGAConfig(population_size=0)
            
    def test_invalid_mutation_rate(self):
        """Teste la validation du taux de mutation."""
        with self.assertRaises(ValidationError):
            NSGAConfig(mutation_rate=1.5)
            
    def test_invalid_crossover_rate(self):
        """Teste la validation du taux de croisement."""
        with self.assertRaises(ValidationError):
            NSGAConfig(crossover_rate=-0.1)

class TestSolution(unittest.TestCase):
    """Tests pour la classe Solution."""
    
    def setUp(self):
        """Initialise les données pour les tests."""
        self.parameters = {
            'param1': 0.5,
            'param2': 1.0
        }
        self.objectives = {
            'return': 0.1,
            'risk': 0.05
        }
        
    def test_valid_solution(self):
        """Teste une solution valide."""
        solution = Solution(
            parameters=self.parameters,
            objectives=self.objectives
        )
        self.assertEqual(solution.parameters, self.parameters)
        self.assertEqual(solution.objectives, self.objectives)
        self.assertEqual(solution.rank, 0)
        
    def test_solution_comparison(self):
        """Teste la comparaison entre solutions."""
        solution1 = Solution(
            parameters=self.parameters,
            objectives={'obj1': 1.0}
        )
        solution2 = Solution(
            parameters=self.parameters,
            objectives={'obj1': 2.0}
        )
        self.assertLess(solution1.objectives['obj1'], solution2.objectives['obj1'])

class TestNSGAII(unittest.TestCase):
    """Tests pour la classe NSGAII."""
    
    def setUp(self):
        """Initialise les données pour les tests."""
        self.objectives = [
            ObjectiveConfig("return", weight=0.6),
            ObjectiveConfig("risk", weight=0.4, minimize=True)
        ]
        
        self.config = NSGAConfig(
            population_size=20,
            generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            tournament_size=3,
            objectives=self.objectives
        )
        
        self.parameter_bounds = {
            'param1': (-1.0, 1.0),
            'param2': (0.0, 2.0)
        }
        
        def evaluation_function(params: Dict[str, float], data: np.ndarray) -> Dict[str, float]:
            """Fonction d'évaluation simple pour les tests."""
            return {
                'return': params['param1'] ** 2,
                'risk': abs(params['param2'])
            }
            
        self.evaluation_function = evaluation_function
        self.data = np.random.randn(100, 5)
        
    def test_initialization(self):
        """Teste l'initialisation de l'algorithme."""
        nsga = NSGAII(self.config)
        self.assertEqual(len(nsga.config.objectives), 2)
        
    def test_population_initialization(self):
        """Teste l'initialisation de la population."""
        nsga = NSGAII(self.config)
        nsga.initialize_population(self.parameter_bounds)
        self.assertEqual(len(nsga.population), self.config.population_size)
        
    def test_non_dominated_sort(self):
        """Teste le tri non-dominé."""
        nsga = NSGAII(self.config)
        nsga.initialize_population(self.parameter_bounds)
        nsga.evaluate_population(self.evaluation_function, self.data)
        nsga.non_dominated_sort()
        self.assertTrue(all(s.rank >= 0 for s in nsga.population))
        
    def test_optimization(self):
        """Teste le processus d'optimisation complet."""
        nsga = NSGAII(self.config)
        solutions = nsga.optimize(
            self.evaluation_function,
            self.data,
            self.parameter_bounds
        )
        self.assertGreater(len(solutions), 0)
        
    def test_time_series_validation(self):
        """Teste la validation croisée temporelle."""
        self.config.time_series_validation = TimeSeriesValidationConfig(
            enabled=True,
            n_splits=3,
            train_size=0.8
        )
        nsga = NSGAII(self.config)
        nsga._setup_time_series_validation(len(self.data))
        self.assertIsNotNone(nsga.time_series_splits)
        
    def test_parallel_evaluation(self):
        """Teste l'évaluation parallèle."""
        self.config.parallel_evaluation = True
        nsga = NSGAII(self.config)
        nsga.initialize_population(self.parameter_bounds)
        nsga.evaluate_population(self.evaluation_function, self.data)
        self.assertTrue(all(s.objectives for s in nsga.population))
        
    def test_adaptive_mutation(self):
        """Teste l'adaptation du taux de mutation."""
        self.config.adaptive_mutation = True
        nsga = NSGAII(self.config)
        solutions = nsga.optimize(
            self.evaluation_function,
            self.data,
            self.parameter_bounds
        )
        self.assertGreater(len(solutions), 0)
        
    def test_invalid_config(self):
        """Teste la gestion des configurations invalides."""
        with self.assertRaises(ValidationError):
            NSGAII(None)
            
    def test_optimization_error(self):
        """Teste la gestion des erreurs d'optimisation."""
        def failing_evaluation(params, data):
            raise Exception("Erreur simulée")
            
        nsga = NSGAII(self.config)
        with self.assertRaises(OptimizationError):
            nsga.optimize(failing_evaluation, self.data, self.parameter_bounds)
            
    def test_results_saving(self):
        """Teste la sauvegarde des résultats."""
        nsga = NSGAII(self.config)
        solutions = nsga.optimize(
            self.evaluation_function,
            self.data,
            self.parameter_bounds
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "results.json"
            
            # Sauvegarde
            summary = nsga.get_optimization_summary()
            with open(save_path, 'w') as f:
                json.dump(summary, f)
                
            # Vérification
            self.assertTrue(save_path.exists())
            with open(save_path, 'r') as f:
                loaded = json.load(f)
                self.assertEqual(
                    loaded['final_pareto_front_size'],
                    len(solutions)
                )

if __name__ == '__main__':
    unittest.main() 