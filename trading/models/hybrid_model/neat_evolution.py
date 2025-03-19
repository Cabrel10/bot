import os
import neat
import numpy as np
from typing import Dict, Any, List
import tensorflow as tf

class NEATEvolution:
    """Implémentation de NEAT (NeuroEvolution of Augmenting Topologies) pour l'évolution d'architectures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize_neat()
        
    def _initialize_neat(self):
        """Initialise la configuration NEAT."""
        # Création du fichier de configuration NEAT temporaire
        config_path = self._create_config_file()
        
        # Chargement de la configuration
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Suppression du fichier temporaire
        if os.path.exists(config_path):
            os.remove(config_path)
            
    def _create_config_file(self) -> str:
        """Crée un fichier de configuration NEAT temporaire."""
        config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.config.get('fitness_threshold', 3.9)}
pop_size             = {self.config.get('population_size', 50)}
reset_on_extinction  = {str(self.config.get('reset_on_extinction', False)).lower()}

[DefaultGenome]
num_inputs              = {self.config.get('num_inputs', 10)}
num_hidden              = {self.config.get('num_hidden', 3)}
num_outputs             = {self.config.get('num_outputs', 1)}
initial_connection      = partial_direct 0.5
feed_forward            = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.6
conn_add_prob          = 0.2
conn_delete_prob       = 0.2
node_add_prob          = 0.2
node_delete_prob       = 0.2
activation_default      = tanh
activation_options     = tanh
activation_mutate_rate = 0.0
aggregation_default    = sum
aggregation_options    = sum
aggregation_mutate_rate = 0.0
bias_init_mean         = 0.0
bias_init_stdev        = 1.0
bias_replace_rate      = 0.1
bias_mutate_rate       = 0.7
bias_mutate_power      = 0.5
bias_max_value         = 30.0
bias_min_value         = -30.0
response_init_mean     = 1.0
response_init_stdev    = 0.0
response_replace_rate  = 0.0
response_mutate_rate   = 0.0
response_mutate_power  = 0.0
response_max_value     = 30.0
response_min_value     = -30.0

weight_max_value        = 30
weight_min_value        = -30
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
weight_mutate_power     = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        
        # Écriture du fichier de configuration
        config_path = "temp_neat_config.txt"
        with open(config_path, "w") as f:
            f.write(config_text)
            
        return config_path
        
    def evolve(self, data: np.ndarray, generations: int = None) -> neat.DefaultGenome:
        """Fait évoluer l'architecture sur les données fournies."""
        pop = neat.Population(self.neat_config)
        
        # Nombre de générations
        n_generations = generations or self.config.get('generations', 10)
        
        # Fonction d'évaluation
        def eval_genome(genome, config):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            return self._evaluate_network(net, data)
            
        # Évolution
        pop.run(eval_genome, n_generations)
        
        return pop.best_genome
        
    def _evaluate_network(self, network, data: np.ndarray) -> float:
        """Évalue un réseau sur les données."""
        try:
            predictions = []
            for x in data:
                output = network.activate(x)
                predictions.append(output[0])
            
            # Calcul du score
            return self._calculate_fitness(predictions, data)
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {str(e)}")
            return 0.0
            
    def _calculate_fitness(self, predictions: List[float], 
                         data: np.ndarray) -> float:
        """Calcule le score de fitness."""
        try:
            predictions = np.array(predictions)
            actual = data[:, -1]  # Suppose que la dernière colonne est la cible
            
            # Métriques de performance
            mse = np.mean((predictions - actual) ** 2)
            correlation = np.corrcoef(predictions, actual)[0, 1]
            
            # Score composite
            return 1.0 / (1.0 + mse) * (1.0 + correlation)
            
        except Exception as e:
            print(f"Erreur dans le calcul du fitness: {str(e)}")
            return 0.0 