from typing import Dict
import pandas as pd

class MetaLearner:
    """Gère le pré-entraînement et le fine-tuning sur différents instruments."""
    
    def __init__(self, config: Dict):
        self.base_models = {}
        self.transfer_params = config['meta_learning']
        self.instrument_features = {}

    def pre_train(self, instruments_data: Dict[str, pd.DataFrame]):
        """Pré-entraînement sur plusieurs instruments."""
        for instrument, data in instruments_data.items():
            self.base_models[instrument] = self._train_base_model(data)
            self.instrument_features[instrument] = self._extract_features(data)

    def fine_tune(self, target_instrument: str, recent_data: pd.DataFrame):
        """Fine-tuning en temps réel pour un instrument spécifique."""
        base_model = self._select_closest_model(target_instrument, recent_data)
        return self._adapt_model(base_model, recent_data) 