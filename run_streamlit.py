import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import and run the Streamlit interface
from src.core.data.data_acquisition import BatchDataAcquisition

if __name__ == "__main__":
    # Créer une instance de BatchDataAcquisition pour l'interface
    data_acquisition = BatchDataAcquisition()
    
    # Lancer Streamlit avec la page principale
    import streamlit.cli as stcli
    sys.argv = ["streamlit", "run", str(project_root / "src" / "interface" / "main_dashboard.py")]
    stcli.main()