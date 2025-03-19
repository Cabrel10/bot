import pytest
from src.genetic.operators import GeneticOperators
import numpy as np

@pytest.fixture
def sample_population():
    return [
        {'rsi_period': 14, 'ma_period': 20, 'threshold': 0.5},
        {'rsi_period': 7, 'ma_period': 50, 'threshold': 0.3},
        {'rsi_period': 21, 'ma_period': 10, 'threshold': 0.7},
        {'rsi_period': 10, 'ma_period': 30, 'threshold': 0.4}
    ]

@pytest.fixture
def param_bounds():
    return {
        'rsi_period': (2, 30),
        'ma_period': (5, 200),
        'threshold': (0.1, 0.9)
    }

def test_selection(sample_population):
    operators = GeneticOperators()
    fitness_scores = [0.5, 0.8, 0.3, 0.6]
    
    parents = operators.select_parents(sample_population, fitness_scores, n_parents=2)
    assert len(parents) == 2
    assert all(isinstance(p, dict) for p in parents)

def test_crossover(sample_population):
    operators = GeneticOperators(crossover_rate=1.0)
    parent1, parent2 = sample_population[0:2]
    
    child1, child2 = operators.crossover(parent1, parent2)
    
    assert child1 != parent1 or child2 != parent2
    assert all(key in child1 for key in parent1.keys())
    assert all(key in child2 for key in parent2.keys())

def test_mutation(sample_population, param_bounds):
    operators = GeneticOperators(mutation_rate=1.0, param_bounds=param_bounds)
    individual = sample_population[0]
    
    mutated = operators.mutate(individual)
    
    assert mutated != individual
    assert all(param_bounds[key][0] <= mutated[key] <= param_bounds[key][1] 
              for key in param_bounds.keys())

def test_next_generation(sample_population):
    operators = GeneticOperators()
    fitness_scores = [0.5, 0.8, 0.3, 0.6]
    
    next_gen = operators.create_next_generation(sample_population, fitness_scores)
    
    assert len(next_gen) == len(sample_population)
    assert all(isinstance(ind, dict) for ind in next_gen) 