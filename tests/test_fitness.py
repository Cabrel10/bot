import pytest
from typing import List
from src.core.models.fitness import FitnessEvaluator
from src.core.position import Position
from datetime import datetime, timedelta

def create_test_positions(profit_pcts: list) -> List[Position]:
    positions = []
    start_time = datetime.now()
    
    for i, pct in enumerate(profit_pcts):
        pos = Position(
            symbol="BTCUSDT",
            entry_price=1000,
            exit_price=1000 * (1 + pct/100),
            entry_time=start_time + timedelta(hours=i),
            exit_time=start_time + timedelta(hours=i+1),
            size=1.0
        )
        positions.append(pos)
    
    return positions

def test_fitness_evaluation():
    evaluator = FitnessEvaluator()
    
    # Création de positions test avec un profit moyen de 2%
    positions = create_test_positions([2] * 25)
    
    score, metrics = evaluator.evaluate(positions, 10000)
    
    assert score > 0
    assert metrics['number_of_trades'] == 25
    assert metrics['total_profit_pct'] > 0
    assert 'sharpe_ratio' in metrics 