"""
Système de sauvegarde et restauration pour le modèle hybride.
"""

import os
import json
import zipfile
import tempfile
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime
import shutil

class ModelPersistence:
    """Gère la sauvegarde et restauration du modèle hybride."""
    
    def __init__(self, base_path: str = "models/checkpoints"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, model, version: str = None, compress: bool = True):
        """
        Sauvegarde le modèle avec compression et versioning.
        
        Args:
            model: Instance du modèle hybride
            version: Version du modèle (utilise timestamp si non spécifié)
            compress: Si True, compresse les fichiers
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        save_path = self.base_path / version
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Sauvegarde du modèle neuronal
            self._save_neural_network(model.nn_model, temp_path)
            
            # 2. Sauvegarde du modèle génétique
            self._save_genetic_algorithm(model.ga_model, temp_path)
            
            # 3. Sauvegarde des métadonnées
            self._save_metadata(model, temp_path, version)
            
            # 4. Compression si demandée
            if compress:
                self._compress_checkpoint(temp_path, save_path / f"model_{version}.zip")
            else:
                shutil.copytree(temp_path, save_path, dirs_exist_ok=True)
                
        return str(save_path)
    
    def restore_model(self, model, version: str = "latest"):
        """
        Restaure un modèle depuis une sauvegarde.
        
        Args:
            model: Instance du modèle hybride à restaurer
            version: Version à restaurer ("latest" pour la plus récente)
        """
        if version == "latest":
            version = self._get_latest_version()
            
        load_path = self.base_path / version
        
        # Décompression si nécessaire
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if (load_path / f"model_{version}.zip").exists():
                self._decompress_checkpoint(load_path / f"model_{version}.zip", temp_path)
                load_path = temp_path
            
            # 1. Restauration du modèle neuronal
            self._restore_neural_network(model.nn_model, load_path)
            
            # 2. Restauration du modèle génétique
            self._restore_genetic_algorithm(model.ga_model, load_path)
            
            # 3. Restauration des métadonnées
            self._restore_metadata(model, load_path)
            
        return model
    
    def _save_neural_network(self, nn_model, path: Path):
        """Sauvegarde optimisée du réseau neuronal."""
        # Sauvegarde du modèle quantifié
        converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open(path / "neural_network.tflite", "wb") as f:
            f.write(tflite_model)
            
        # Sauvegarde des poids séparément pour restauration rapide
        nn_model.save_weights(str(path / "neural_network_weights"))
    
    def _save_genetic_algorithm(self, ga_model, path: Path):
        """Sauvegarde du modèle génétique."""
        ga_state = {
            "population": ga_model.get_population_state(),
            "best_individual": ga_model.get_best_individual(),
            "generation": ga_model.current_generation,
            "fitness_history": ga_model.fitness_history
        }
        
        with open(path / "genetic_algorithm.json", "w") as f:
            json.dump(ga_state, f)
    
    def _save_metadata(self, model, path: Path, version: str):
        """Sauvegarde des métadonnées du modèle."""
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "model_type": "hybrid",
            "hyperparameters": model.get_hyperparameters(),
            "performance_metrics": model.evaluate(None),
            "feature_set": model.get_features().to_dict()
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def _compress_checkpoint(self, source_path: Path, target_path: Path):
        """Compresse les fichiers de checkpoint."""
        with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in source_path.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(source_path))
    
    def _decompress_checkpoint(self, source_path: Path, target_path: Path):
        """Décompresse les fichiers de checkpoint."""
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(target_path)
    
    def _restore_neural_network(self, nn_model, path: Path):
        """Restaure le réseau neuronal."""
        nn_model.load_weights(str(path / "neural_network_weights"))
    
    def _restore_genetic_algorithm(self, ga_model, path: Path):
        """Restaure le modèle génétique."""
        with open(path / "genetic_algorithm.json", "r") as f:
            ga_state = json.load(f)
            
        ga_model.restore_population_state(ga_state["population"])
        ga_model.set_best_individual(ga_state["best_individual"])
        ga_model.current_generation = ga_state["generation"]
        ga_model.fitness_history = ga_state["fitness_history"]
    
    def _restore_metadata(self, model, path: Path):
        """Restaure les métadonnées du modèle."""
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        model.metadata = metadata
    
    def _get_latest_version(self) -> str:
        """Récupère la version la plus récente."""
        versions = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        if not versions:
            raise ValueError("Aucune version trouvée")
        return sorted(versions)[-1]
    
    def list_versions(self):
        """Liste toutes les versions disponibles."""
        versions = []
        for version_dir in self.base_path.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    versions.append({
                        "version": version_dir.name,
                        "timestamp": metadata["timestamp"],
                        "metrics": metadata["performance_metrics"]
                    })
        return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
    
    def cleanup_old_versions(self, keep_last_n: int = 5):
        """Nettoie les anciennes versions en gardant les n plus récentes."""
        versions = self.list_versions()
        if len(versions) > keep_last_n:
            for version in versions[keep_last_n:]:
                version_path = self.base_path / version["version"]
                if version_path.exists():
                    shutil.rmtree(version_path) 