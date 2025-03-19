from .model_interface import ModelInterface

class EnsembleModel(ModelInterface):
    """Modèle d'ensemble pour combiner plusieurs modèles."""
    
    def combine_predictions(self, predictions):
        """Combine les prédictions de plusieurs modèles."""
        pass 