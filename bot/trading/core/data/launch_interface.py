"""
Point d'entrée pour l'interface Streamlit du bot de trading.
Ce fichier redirige vers l'interface principale dans src/interface/main_dashboard.py.
"""

import os
import sys
from pathlib import Path

# Définition du chemin racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
print(f"Chemin racine du projet: {PROJECT_ROOT}")

# Ajout du chemin racine au PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Point d'entrée principal de l'interface."""
    try:
        # Configuration de l'environnement
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
        
        # Lancement de l'interface Streamlit
        import streamlit.web.cli as stcli
        
        sys.argv = [
            "streamlit",
            "run",
            str(PROJECT_ROOT / "src" / "interface" / "main_dashboard.py"),
            "--server.port=8501",
            "--server.address=localhost"
        ]
        
        stcli.main()
        
    except Exception as e:
        print(f"Erreur lors du lancement de l'interface: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
