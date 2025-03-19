"""Gestionnaire des modèles de trading."""
import os
from datetime import datetime
import tensorflow as tf
from trading.utils.logging import TradingLogger
from trading.core.models import NeuralNetwork, GeneticAlgorithm

class ModelManager:
    """Gestionnaire des modèles de trading."""
    
    def __init__(self):
        self.logger = TradingLogger()
        self.models_dir = '/app/models'
        self.logs_dir = '/app/logs'
        self.setup_directories()
        
        # Initialisation des modèles
        self.neural_net = NeuralNetwork()
        self.genetic_algo = GeneticAlgorithm()
        
        # Configuration TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(self.logs_dir, 'train', current_time)
        self.test_log_dir = os.path.join(self.logs_dir, 'test', current_time)
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
    
    def setup_directories(self):
        """Création des répertoires nécessaires."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def train_models(self, data):
        """Entraînement des modèles."""
        try:
            # Entraînement du réseau de neurones
            with self.train_summary_writer.as_default():
                metrics = self.neural_net.train(data)
                for name, value in metrics.items():
                    tf.summary.scalar(name, value, step=0)
            
            # Optimisation génétique
            self.genetic_algo.optimize(data)
            
            # Sauvegarde des modèles
            self.save_models()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement des modèles: {e}")
    
    def save_models(self):
        """Sauvegarde des modèles entraînés."""
        try:
            model_path = os.path.join(self.models_dir, f'model_{datetime.now():%Y%m%d_%H%M%S}')
            self.neural_net.save(f"{model_path}_nn")
            self.genetic_algo.save(f"{model_path}_ga")
            self.logger.info("Modèles sauvegardés avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")

if __name__ == '__main__':
    model_manager = ModelManager()
    # Boucle principale du service
    while True:
        try:
            # TODO: Implémenter la logique de récupération des données
            # et d'entraînement périodique des modèles
            pass
        except Exception as e:
            model_manager.logger.error(f"Erreur dans la boucle principale: {e}")
            import time
            time.sleep(60)  # Attente avant nouvelle tentative
