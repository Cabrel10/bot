"""
Optimiseur pour le modèle hybride.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from ..common.model_interface import ModelInterface
from ..neural_network_model.model import NeuralNetworkModel
from .params import OptimizationParams, NeuralNetworkParams

class HybridOptimizer:
    """
    Optimiseur qui utilise l'algorithme génétique pour optimiser
    les hyperparamètres du réseau neuronal.
    """

    def __init__(self):
        """Initialise l'optimiseur."""
        self.best_params = None
        self.best_score = float('-inf')
        self.population = []
        self.scores = []
        self.generation = 0
        self.history = []

    def optimize(
        self,
        model: NeuralNetworkModel,
        data: pd.DataFrame,
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """
        Optimise les hyperparamètres du modèle.

        Args:
            model: Modèle à optimiser
            data: Données d'entraînement
            params: Paramètres d'optimisation

        Returns:
            Dict[str, Any]: Meilleurs hyperparamètres trouvés
        """
        # Préparation des données
        X_train, X_val, y_train, y_val = self._prepare_data(data)

        # Initialisation de la population
        self.population = self._initialize_population(params)
        
        # Boucle principale d'optimisation
        for generation in range(params.generations):
            self.generation = generation
            
            # Évaluation de la population
            self.scores = self._evaluate_population(
                self.population,
                model,
                X_train, y_train,
                X_val, y_val
            )
            
            # Mise à jour du meilleur score
            best_idx = np.argmax(self.scores)
            if self.scores[best_idx] > self.best_score:
                self.best_score = self.scores[best_idx]
                self.best_params = self.population[best_idx]
                
            # Enregistrement de l'historique
            self.history.append({
                'generation': generation,
                'best_score': self.best_score,
                'mean_score': np.mean(self.scores),
                'std_score': np.std(self.scores)
            })
            
            # Vérification de la convergence
            if self._check_convergence(params):
                break
                
            # Création de la nouvelle génération
            self.population = self._create_next_generation(
                self.population,
                self.scores,
                params
            )
            
        return self._decode_parameters(self.best_params)

    def _prepare_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'optimisation.

        Args:
            data: DataFrame contenant les données

        Returns:
            Tuple contenant les données d'entraînement et de validation
        """
        # Séparation features/target
        X = data.drop('target', axis=1).values
        y = data['target'].values
        
        # Split train/validation
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _initialize_population(
        self,
        params: OptimizationParams
    ) -> List[Dict[str, Any]]:
        """
        Initialise la population avec des paramètres aléatoires.

        Args:
            params: Paramètres d'optimisation

        Returns:
            List[Dict[str, Any]]: Population initiale
        """
        population = []
        for _ in range(params.population_size):
            # Génération de paramètres aléatoires
            individual = {
                'learning_rate': np.random.uniform(1e-4, 1e-2),
                'batch_size': np.random.choice([16, 32, 64, 128]),
                'dropout_rate': np.random.uniform(0.1, 0.5),
                'hidden_layers': self._generate_random_architecture(),
                'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
                'activation': np.random.choice(['relu', 'tanh', 'elu'])
            }
            population.append(individual)
        return population

    def _generate_random_architecture(self) -> List[int]:
        """
        Génère une architecture aléatoire pour le réseau neuronal.

        Returns:
            List[int]: Liste des tailles des couches cachées
        """
        n_layers = np.random.randint(2, 5)
        architecture = []
        current_size = 128
        for _ in range(n_layers):
            architecture.append(current_size)
            current_size = current_size // 2
        return architecture

    def _evaluate_population(
        self,
        population: List[Dict[str, Any]],
        model: NeuralNetworkModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> List[float]:
        """
        Évalue chaque individu de la population.

        Args:
            population: Liste des individus à évaluer
            model: Modèle à optimiser
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation

        Returns:
            List[float]: Scores de chaque individu
        """
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(
                lambda params: self._evaluate_individual(
                    params, model, X_train, y_train, X_val, y_val
                ),
                population
            ))
        return scores

    def _evaluate_individual(
        self,
        params: Dict[str, Any],
        model: NeuralNetworkModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Évalue un individu spécifique.

        Args:
            params: Paramètres à évaluer
            model: Modèle à optimiser
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            X_val: Données de validation
            y_val: Labels de validation

        Returns:
            float: Score de l'individu
        """
        try:
            # Création d'une copie du modèle avec les nouveaux paramètres
            model_copy = NeuralNetworkModel()
            model_copy.update_parameters(params)
            
            # Entraînement rapide
            history = model_copy.train(
                pd.DataFrame(X_train),
                epochs=10,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Retourne le meilleur score de validation
            return max(history.history['val_accuracy'])
        except Exception:
            return float('-inf')

    def _create_next_generation(
        self,
        population: List[Dict[str, Any]],
        scores: List[float],
        params: OptimizationParams
    ) -> List[Dict[str, Any]]:
        """
        Crée la prochaine génération d'individus.

        Args:
            population: Population actuelle
            scores: Scores de la population
            params: Paramètres d'optimisation

        Returns:
            List[Dict[str, Any]]: Nouvelle génération
        """
        new_population = []
        
        # Élitisme
        elite_idx = np.argsort(scores)[-params.elite_size:]
        new_population.extend([population[i] for i in elite_idx])
        
        # Création du reste de la population
        while len(new_population) < params.population_size:
            if np.random.random() < params.crossover_rate:
                # Crossover
                parent1 = self._select_parent(population, scores, params)
                parent2 = self._select_parent(population, scores, params)
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._select_parent(population, scores, params)
                child = self._mutate(parent.copy(), params)
            
            new_population.append(child)
            
        return new_population

    def _select_parent(
        self,
        population: List[Dict[str, Any]],
        scores: List[float],
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """
        Sélectionne un parent pour la reproduction.

        Args:
            population: Population actuelle
            scores: Scores de la population
            params: Paramètres d'optimisation

        Returns:
            Dict[str, Any]: Parent sélectionné
        """
        # Sélection par tournoi
        tournament_idx = np.random.choice(
            len(population),
            size=params.tournament_size,
            replace=False
        )
        tournament_scores = [scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_scores)]
        return population[winner_idx]

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Effectue un croisement entre deux parents.

        Args:
            parent1: Premier parent
            parent2: Second parent

        Returns:
            Dict[str, Any]: Enfant résultant du croisement
        """
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], (int, float)):
                # Croisement arithmétique pour les valeurs numériques
                alpha = np.random.random()
                child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
            elif isinstance(parent1[key], list):
                # Croisement des architectures
                crossover_point = np.random.randint(1, min(len(parent1[key]), len(parent2[key])))
                child[key] = parent1[key][:crossover_point] + parent2[key][crossover_point:]
            else:
                # Pour les autres types, choix aléatoire entre les deux parents
                child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
        return child

    def _mutate(
        self,
        individual: Dict[str, Any],
        params: OptimizationParams
    ) -> Dict[str, Any]:
        """
        Applique une mutation à un individu.

        Args:
            individual: Individu à muter
            params: Paramètres d'optimisation

        Returns:
            Dict[str, Any]: Individu muté
        """
        for key in individual.keys():
            if np.random.random() < params.mutation_rate:
                if key == 'learning_rate':
                    individual[key] *= np.random.uniform(0.5, 2.0)
                elif key == 'batch_size':
                    individual[key] = np.random.choice([16, 32, 64, 128])
                elif key == 'dropout_rate':
                    individual[key] = np.clip(
                        individual[key] + np.random.normal(0, 0.1),
                        0.1, 0.5
                    )
                elif key == 'hidden_layers':
                    if np.random.random() < 0.5:
                        # Ajoute ou retire une couche
                        if len(individual[key]) > 2 and np.random.random() < 0.5:
                            individual[key].pop()
                        else:
                            individual[key].append(individual[key][-1] // 2)
                    else:
                        # Modifie la taille d'une couche
                        idx = np.random.randint(len(individual[key]))
                        individual[key][idx] = int(
                            individual[key][idx] * np.random.uniform(0.5, 2.0)
                        )
                elif key in ['optimizer', 'activation']:
                    options = {
                        'optimizer': ['adam', 'rmsprop', 'sgd'],
                        'activation': ['relu', 'tanh', 'elu']
                    }
                    individual[key] = np.random.choice(options[key])
        return individual

    def _check_convergence(self, params: OptimizationParams) -> bool:
        """
        Vérifie si l'optimisation a convergé.

        Args:
            params: Paramètres d'optimisation

        Returns:
            bool: True si l'optimisation a convergé
        """
        if len(self.history) < params.max_iterations_without_improvement:
            return False
            
        recent_scores = [h['best_score'] for h in self.history[-params.max_iterations_without_improvement:]]
        score_improvement = max(recent_scores) - min(recent_scores)
        
        return score_improvement < params.convergence_threshold

    def _decode_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Décode les paramètres optimisés en format utilisable par le modèle.

        Args:
            params: Paramètres à décoder

        Returns:
            Dict[str, Any]: Paramètres décodés
        """
        return {
            'learning_rate': float(params['learning_rate']),
            'batch_size': int(params['batch_size']),
            'dropout_rate': float(params['dropout_rate']),
            'hidden_layers': [int(x) for x in params['hidden_layers']],
            'optimizer': str(params['optimizer']),
            'activation': str(params['activation'])
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Retourne l'historique d'optimisation.

        Returns:
            List[Dict[str, Any]]: Historique d'optimisation
        """
        return self.history.copy()

    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Retourne les meilleurs paramètres trouvés.

        Returns:
            Optional[Dict[str, Any]]: Meilleurs paramètres
        """
        return self._decode_parameters(self.best_params) if self.best_params else None
