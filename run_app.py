#!/usr/bin/env python3
"""
Script principal pour lancer l'application de trading.
"""

import os
import sys
from pathlib import Path

# Définition du chemin racine du projet
PROJECT_ROOT = Path(__file__).parent.absolute()
print(f"Chemin racine du projet: {PROJECT_ROOT}")

# Ajout du chemin racine au PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def setup_directories():
    """Crée les répertoires nécessaires pour l'application."""
    directories = [
        "data/raw",
        "data/processed",
        "data/datasets",
        "models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"Répertoire créé/vérifié: {path}")

def main():
    """Point d'entrée principal de l'application."""
    try:
        # Création des répertoires nécessaires
        setup_directories()
        
        # Configuration de l'environnement
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
        
        # Lancement de l'interface Streamlit
        import streamlit.web.cli as stcli
        
        sys.argv = [
            "streamlit",
            "run",
            str(PROJECT_ROOT / "bot/trading/core/data/launch_interface.py"),
            "--server.port=8501",
            "--server.address=localhost"
        ]
        
        sys.exit(stcli.main())
        
    except Exception as e:
        print(f"Erreur lors du lancement de l'application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 