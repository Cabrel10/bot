# Hybrid Trading Model

## Overview
This hybrid model combines the power of genetic algorithms and neural networks for optimized trading predictions. The genetic algorithm component is used to optimize the hyperparameters of the neural network, creating a self-tuning system that adapts to market conditions.

## Structure
The model consists of four main components:

1. **HybridModel** (`model.py`)
   - Main model implementation
   - Combines genetic algorithm and neural network predictions
   - Implements both ModelInterface and EnsembleModel protocols
   - Handles model training, prediction, and evaluation

2. **HybridOptimizer** (`optimizer.py`)
   - Optimizes neural network hyperparameters using genetic algorithms
   - Implements population-based optimization
   - Supports parallel evaluation of candidates
   - Includes elitism and adaptive mutation rates

3. **Parameters** (`params.py`)
   - Defines all configurable parameters
   - Includes optimization, neural network, and genetic algorithm parameters
   - Supports JSON serialization for easy configuration management

4. **Test Suite** (`test_hybrid.py`)
   - Provides comprehensive testing of the hybrid model
   - Includes synthetic data generation
   - Tests all major functionalities

## Dependencies
Required packages are listed in `requirements/hybrid_model.txt`:
- numpy (≥1.21.0)
- pandas (≥1.3.0)
- scikit-learn (≥1.0.0)
- tensorflow (≥2.8.0)

## Usage

### Basic Usage
```python
from trading.models.hybrid_model import HybridModel, HybridModelParams

# Initialize model with default parameters
model = HybridModel()

# Or with custom parameters
params = HybridModelParams(
    neural_params=NeuralNetworkParams(
        learning_rate=0.001,
        batch_size=32
    ),
    genetic_params=GeneticParams(
        population_size=50,
        generations=100
    )
)
model = HybridModel(params)

# Train the model
model.train(data)

# Make predictions
predictions = model.predict(new_data)
```

### Advanced Configuration
The model supports extensive configuration through its parameter classes:

```python
from trading.models.hybrid_model.params import (
    OptimizationParams,
    NeuralNetworkParams,
    GeneticParams
)

# Customize optimization parameters
opt_params = OptimizationParams(
    population_size=100,
    generations=200,
    mutation_rate=0.1,
    crossover_rate=0.8
)

# Customize neural network architecture
nn_params = NeuralNetworkParams(
    learning_rate=0.001,
    batch_size=32,
    hidden_layers=[64, 32, 16],
    dropout_rate=0.2
)

# Customize genetic algorithm parameters
ga_params = GeneticParams(
    chromosome_length=20,
    fitness_threshold=0.95,
    adaptive_mutation=True
)

# Create model with custom parameters
model = HybridModel(
    HybridModelParams(
        optimization_params=opt_params,
        neural_params=nn_params,
        genetic_params=ga_params
    )
)
```

## Features

### Adaptive Optimization
- Dynamic hyperparameter optimization using genetic algorithms
- Automatic architecture search for neural networks
- Population-based training with elitism
- Convergence detection and early stopping

### Ensemble Capabilities
- Weighted combination of multiple models
- Support for adding custom models to the ensemble
- Dynamic weight adjustment based on performance
- Flexible ensemble methods (weighted average, voting, etc.)

### Model Management
- Save and load model states
- Comprehensive training history
- Performance metrics tracking
- Model validation and diagnostics

## Testing
Run the test suite to verify the model's functionality:

```python
python -m trading.models.hybrid_model.test_hybrid
```

## Performance Metrics
The model tracks various performance metrics:
- Prediction accuracy
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

## Best Practices
1. **Data Preparation**
   - Normalize/standardize input features
   - Handle missing values appropriately
   - Ensure sufficient training data

2. **Model Configuration**
   - Start with default parameters
   - Adjust based on your specific use case
   - Monitor optimization progress

3. **Production Deployment**
   - Implement proper error handling
   - Monitor model performance
   - Regularly update and retrain

## Contributing
When contributing to this model:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Maintain type hints and docstrings
