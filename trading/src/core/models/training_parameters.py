from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TemporalParameters:
    """Gestion des paramètres temporels pour l'entraînement des modèles."""
    
    def __init__(self, 
                 window_size: int = 60,
                 update_frequency: str = '1D',
                 prediction_horizon: int = 5,
                 train_test_split: float = 0.8):
        """
        Initialise les paramètres temporels.
        
        Args:
            window_size: Taille de la fenêtre glissante (en périodes)
            update_frequency: Fréquence de mise à jour ('1D', '1H', etc.)
            prediction_horizon: Horizon de prédiction (en périodes)
            train_test_split: Ratio de division train/test
            
        Raises:
            ValueError: Si les paramètres sont invalides
        """
        # Validation des paramètres
        if window_size <= 0:
            raise ValueError("La taille de la fenêtre doit être positive")
        if prediction_horizon <= 0:
            raise ValueError("L'horizon de prédiction doit être positif")
        if train_test_split <= 0 or train_test_split >= 1:
            raise ValueError("Le ratio de division doit être entre 0 et 1 exclusivement")
            
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.prediction_horizon = prediction_horizon
        self.train_test_split = train_test_split
    
    def create_rolling_windows(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des fenêtres glissantes pour l'entraînement.
        
        Args:
            data: DataFrame avec index temporel
            
        Returns:
            X: Features (n_samples, window_size, n_features)
            y: Labels (n_samples, prediction_horizon)
            
        Raises:
            ValueError: Si les données sont vides, invalides ou contiennent des valeurs manquantes
        """
        # Validation des données
        if data.empty:
            raise ValueError("Les données ne peuvent pas être vides")
        if len(data) < self.window_size + self.prediction_horizon:
            raise ValueError("Les données sont trop courtes pour la taille de fenêtre et l'horizon spécifiés")
        if data.isna().any().any():
            raise ValueError("Les données contiennent des valeurs manquantes")
            
        n_features = data.shape[1]
        n_samples = len(data) - self.window_size - self.prediction_horizon + 1
        
        X = np.zeros((n_samples, self.window_size, n_features))
        y = np.zeros((n_samples, self.prediction_horizon))
        
        for i in range(n_samples):
            X[i] = data.iloc[i:i+self.window_size].values
            y[i] = data.iloc[i+self.window_size:i+self.window_size+self.prediction_horizon].values[:,0]
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divise les données en ensembles d'entraînement et de test.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * self.train_test_split)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def create_time_windows(self, start_date: datetime, 
                          end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Crée des fenêtres temporelles pour l'évaluation.
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Liste de tuples (début, fin) pour chaque fenêtre
        """
        # Utiliser pd.date_range pour générer les dates de début de fenêtre
        # Cela garantit que le nombre de fenêtres correspond exactement à l'attendu
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.update_frequency)
        windows = []
        
        for start in date_range:
            window_end = min(start + timedelta(days=self.window_size), end_date)
            windows.append((start, window_end))
        
        return windows
    
    def evaluate_window_sizes(self, data: pd.DataFrame, 
                            model: object,
                            window_sizes: List[int]) -> Dict[int, float]:
        """
        Évalue différentes tailles de fenêtres.
        
        Args:
            data: DataFrame avec index temporel
            model: Modèle à évaluer
            window_sizes: Liste des tailles de fenêtres à tester
            
        Returns:
            Dictionnaire {taille_fenetre: performance}
        """
        results = {}
        original_size = self.window_size
        
        for size in window_sizes:
            self.window_size = size
            X, y = self.create_rolling_windows(data)
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            model.train(X_train, y_train)
            performance = model.evaluate(X_test, y_test)
            results[size] = performance
        
        self.window_size = original_size
        return results
    
    def optimize_parameters(self, data: pd.DataFrame,
                          model: object,
                          param_ranges: Dict[str, List[Union[int, str]]]) -> Dict[str, Union[int, str]]:
        """
        Optimise les paramètres temporels.
        
        Args:
            data: DataFrame avec index temporel
            model: Modèle à optimiser
            param_ranges: Dictionnaire des plages de paramètres à tester
            
        Returns:
            Meilleurs paramètres
        """
        best_params = {}
        best_performance = float('-inf')
        
        # Sauvegarde des paramètres originaux
        original_params = {
            'window_size': self.window_size,
            'update_frequency': self.update_frequency,
            'prediction_horizon': self.prediction_horizon
        }
        
        # Test de toutes les combinaisons
        for w_size in param_ranges.get('window_size', [self.window_size]):
            for freq in param_ranges.get('update_frequency', [self.update_frequency]):
                for horizon in param_ranges.get('prediction_horizon', [self.prediction_horizon]):
                    # Mise à jour des paramètres
                    self.window_size = w_size
                    self.update_frequency = freq
                    self.prediction_horizon = horizon
                    
                    # Évaluation
                    X, y = self.create_rolling_windows(data)
                    X_train, X_test, y_train, y_test = self.split_data(X, y)
                    
                    model.train(X_train, y_train)
                    performance = model.evaluate(X_test, y_test)
                    
                    # Mise à jour des meilleurs paramètres
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'window_size': w_size,
                            'update_frequency': freq,
                            'prediction_horizon': horizon
                        }
        
        # Restauration des paramètres originaux
        self.__dict__.update(original_params)
        
        return best_params