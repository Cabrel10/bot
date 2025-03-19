#!/usr/bin/env python3
"""
Script de lancement du service de données.
"""

import os
import sys
from pathlib import Path

# Ajout du chemin racine au PYTHONPATH
root_path = str(Path(__file__).parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from trading.core.data.web_interface import DataCollectionUI

def main():
    """Point d'entrée principal du service de données."""
    ui = DataCollectionUI()
    ui.run()

if __name__ == "__main__":
    main() 