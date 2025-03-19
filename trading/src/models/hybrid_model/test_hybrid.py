"""
Script de test pour le modèle hybride.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

from .model import HybridModel
from .params import HybridModelParams

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Génère des données synthétiques pour le test."""
    
    # Génération des features et target
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Création d'un DataFrame
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    df['timestamp'] = dates
    df.set_index('timestamp', inplace=True)
    
    # Normalisation des features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    return df

def test_hybrid_model():
    """Test principal du modèle hybride."""
    
    print("1. Génération des données de test...")
    data = generate_sample_data()
    
    print("2. Initialisation du modèle hybride...")
    model = HybridModel()
    
    print("3. Validation des données...")
    validation_result = model.validate(data)
    print(f"Validation réussie: {validation_result.is_valid}")
    if not validation_result.is_valid:
        print("Erreurs:", validation_result.errors)
        return
    
    print("4. Entraînement du modèle...")
    try:
        model.train(data)
        print("Entraînement terminé avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        return
    
    print("5. Test des prédictions...")
    try:
        predictions = model.predict(data.iloc[-10:].drop('target', axis=1))
        print("Prédictions:", predictions)
    except Exception as e:
        print(f"Erreur lors des prédictions: {str(e)}")
        return
    
    print("6. Évaluation des performances...")
    metrics = model.evaluate(data)
    print("Métriques:", metrics)
    
    print("7. Test de sauvegarde et chargement...")
    try:
        # Sauvegarde
        model.save('test_model')
        print("Modèle sauvegardé")
        
        # Chargement
        new_model = HybridModel()
        new_model.load('test_model')
        print("Modèle chargé")
        
        # Vérification
        new_predictions = new_model.predict(data.iloc[-10:].drop('target', axis=1))
        print("Prédictions après chargement:", new_predictions)
        print("Différence moyenne avec les prédictions originales:", 
              np.mean(np.abs(predictions - new_predictions)))
    except Exception as e:
        print(f"Erreur lors de la sauvegarde/chargement: {str(e)}")
        return
    
    print("\nTest terminé avec succès!")

if __name__ == "__main__":
    test_hybrid_model()
