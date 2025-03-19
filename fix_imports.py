#!/usr/bin/env python3
"""
Script pour corriger les imports dans les tests.
Ce script remplace les imports 'trading.' par 'src.' dans tous les fichiers de test.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Remplace les imports 'trading.' par 'src.' dans un fichier."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remplacer les imports
    modified_content = re.sub(r'from\s+trading\.', 'from src.', content)
    modified_content = re.sub(r'import\s+trading\.', 'import src.', modified_content)
    
    # Écrire le contenu modifié
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    return content != modified_content

def fix_imports_in_directory(directory):
    """Parcourt un répertoire et corrige les imports dans tous les fichiers Python."""
    modified_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    modified_files.append(file_path)
    
    return modified_files

if __name__ == '__main__':
    # Chemin du répertoire des tests
    project_root = Path(__file__).parent
    tests_dir = project_root / 'tests'
    
    # Corriger les imports dans les tests
    modified_files = fix_imports_in_directory(tests_dir)
    
    # Afficher les fichiers modifiés
    print(f"Nombre de fichiers modifiés: {len(modified_files)}")
    for file in modified_files:
        print(f"  - {file}")
