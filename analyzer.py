#!/usr/bin/env python3
import os
import ast
import argparse
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import importlib.util
from pathlib import Path
from collections import defaultdict
from colorama import Fore, Style  # Pour l'utilisation des couleurs

class PythonCodeAnalyzer:
    """
    Analyseur récursif de code Python qui détecte les fonctions, classes et méthodes.
    """
    
    def __init__(self, root_dir, path, output_file, include_complexity=False, include_imports=False, exclude_pattern=None, include_dependencies=False):
        self.root_dir = root_dir
        self.path = path
        self.output_file = output_file
        self.include_complexity = include_complexity
        self.include_imports = include_imports
        self.exclude_pattern = exclude_pattern
        self.include_dependencies = include_dependencies
        
    def analyze(self) -> None:
        """
        Analyse récursivement tous les fichiers Python dans le répertoire spécifié.
        """
        self.results = {}  # Initialiser l'attribut results
        self.ignore_dirs = set(['.venv', 'lib'])  # Ajouter les noms de répertoires à ignorer
        self.ignore_files = set()  # Définir les fichiers à ignorer

        print(f"Analyse du répertoire : {self.root_dir}")  # Log pour vérifier

        for root, dirs, files in os.walk(self.root_dir):
            # Ignorer les répertoires spécifiés
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    print(f"Analyse du fichier : {rel_path}")  # Log pour vérifier
                    try:
                        self.analyze_file(file_path, rel_path)
                    except Exception as e:
                        print(f"Erreur lors de l'analyse de {rel_path}: {e}")
        
        # Vérifier que self.results contient des données
        print(f"Résultats de l'analyse : {self.results}")  # Log pour vérifier
        
        # Analyser les dépendances entre modules si demandé
        if self.include_dependencies:
            self.analyze_dependencies()
    
    def analyze_file(self, file_path: str, rel_path: str) -> None:
        """
        Analyse un fichier Python à la recherche de fonctions, classes et méthodes.
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            try:
                content = f.read()
                tree = ast.parse(content, filename=file_path)
                
                functions = []
                classes = []
                imports = []
                global_vars = []
                
                # Recherche des importations si demandé
                if self.include_imports:
                    imports = self._get_imports(tree)
                    print(f"Importations trouvées dans {rel_path} : {imports}")  # Log pour vérifier
                
                # Recherche des variables globales
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                global_vars.append({
                                    'name': target.id,
                                    'lineno': node.lineno
                                })
                                print(f"Variable globale trouvée : {target.id}")  # Log pour vérifier
                
                for node in ast.iter_child_nodes(tree):
                    # Recherche de fonctions au niveau du module
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_complexity(node) if self.include_complexity else None
                        functions.append({
                            'name': node.name,
                            'lineno': node.lineno,
                            'args': self._get_function_args(node),
                            'docstring': ast.get_docstring(node),
                            'complexity': complexity,
                            'return_type': self._get_return_annotation(node)
                        })
                        print(f"Fonction trouvée : {node.name}")  # Log pour vérifier
                    
                    # Recherche de classes
                    elif isinstance(node, ast.ClassDef):
                        methods = []
                        class_vars = []
                        
                        # Recherche des méthodes et variables de classe
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                complexity = self._calculate_complexity(item) if self.include_complexity else None
                                methods.append({
                                    'name': item.name,
                                    'lineno': item.lineno,
                                    'args': self._get_function_args(item),
                                    'docstring': ast.get_docstring(item),
                                    'complexity': complexity,
                                    'return_type': self._get_return_annotation(item)
                                })
                                print(f"Méthode trouvée : {item.name}")  # Log pour vérifier
                            elif isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name):
                                        class_vars.append({
                                            'name': target.id,
                                            'lineno': item.lineno
                                        })
                                        print(f"Variable de classe trouvée : {target.id}")  # Log pour vérifier
                        
                        # Recherche des classes parentes
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                bases.append(self._get_attribute_name(base))
                        
                        classes.append({
                            'name': node.name,
                            'lineno': node.lineno,
                            'methods': methods,
                            'class_vars': class_vars,
                            'bases': bases,
                            'docstring': ast.get_docstring(node)
                        })
                        print(f"Classe trouvée : {node.name}")  # Log pour vérifier
                
                self.results[rel_path] = {
                    'functions': functions,
                    'classes': classes,
                    'imports': imports,
                    'global_vars': global_vars,
                    'loc': len(content.split('\n')),
                    'path': file_path
                }
                
                # Analyse des définitions redondantes et des problèmes d'indentation
                self.analyze_code(file_path)
                
            except SyntaxError as e:
                print(f"Erreur de syntaxe dans {rel_path}: {e}")
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """
        Obtient le nom complet d'un attribut (e.g., module.submodule.Class)
        """
        if isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        elif isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return node.attr
    
    def _get_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Extrait les importations du fichier.
        """
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'module': name.name,
                        'alias': name.asname,
                        'lineno': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append({
                        'module': f"{module}.{name.name}" if module else name.name,
                        'alias': name.asname,
                        'lineno': node.lineno,
                        'type': 'from'
                    })
        return imports
    
    def _get_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extrait les arguments d'une fonction ou méthode avec leurs annotations de type.
        """
        args = []
        
        # Traitement des arguments positionnels
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_type = self._get_attribute_name(arg.annotation)
                elif isinstance(arg.annotation, ast.Subscript):
                    # Pour les types génériques comme List[int]
                    if isinstance(arg.annotation.value, ast.Name):
                        base_type = arg.annotation.value.id
                        arg_type = f"{base_type}[...]"
                    elif isinstance(arg.annotation.value, ast.Attribute):
                        base_type = self._get_attribute_name(arg.annotation.value)
                        arg_type = f"{base_type}[...]"
            
            args.append({
                'name': arg_name,
                'type': arg_type
            })
        
        # Traitement des arguments avec valeur par défaut
        if node.args.kwonlyargs:
            for kwarg in node.args.kwonlyargs:
                arg_name = kwarg.arg
                arg_type = None
                if kwarg.annotation:
                    if isinstance(kwarg.annotation, ast.Name):
                        arg_type = kwarg.annotation.id
                
                args.append({
                    'name': arg_name,
                    'type': arg_type,
                    'default': True
                })
        
        # Arguments *args et **kwargs
        if node.args.vararg:
            args.append({
                'name': f"*{node.args.vararg.arg}",
                'type': None
            })
        if node.args.kwarg:
            args.append({
                'name': f"**{node.args.kwarg.arg}",
                'type': None
            })
        
        return args
    
    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """
        Extrait l'annotation de type de retour d'une fonction.
        """
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return self._get_attribute_name(node.returns)
            elif isinstance(node.returns, ast.Subscript):
                if isinstance(node.returns.value, ast.Name):
                    return f"{node.returns.value.id}[...]"
                elif isinstance(node.returns.value, ast.Attribute):
                    return f"{self._get_attribute_name(node.returns.value)}[...]"
        return None
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calcule une approximation de la complexité cyclomatique d'une fonction.
        """
        complexity = 1  # Base complexity
        
        # Comptage des structures de contrôle
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.Or):
                complexity += len(child.values) - 1
        
        return complexity
    
    def analyze_dependencies(self) -> None:
        """
        Analyse les dépendances entre les modules du projet.
        """
        for file_path, file_data in self.results.items():
            module_name = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
            dependencies = set()
            
            # Analyse des importations
            for import_info in file_data.get('imports', []):
                import_module = import_info['module'].split('.')[0]
                
                # Vérifier si c'est un module interne au projet
                for other_path in self.results.keys():
                    other_module = other_path.replace('/', '.').replace('\\', '.').replace('.py', '')
                    if other_module == import_module or other_module.endswith(f".{import_module}"):
                        dependencies.add(other_module)
                        break
            
            self.module_dependencies[module_name] = list(dependencies)
    
    def save_results(self) -> None:
        """
        Sauvegarde les résultats dans un fichier dans le format spécifié.
        """
        if self.format == 'json':
            self._save_json()
        elif self.format == 'markdown':
            self._save_markdown()
        elif self.format == 'html':
            self._save_html()
        else:
            print(f"Format {self.format} non supporté. Utilisation du format Markdown par défaut.")
            self._save_markdown()
    
    def _save_json(self) -> None:
        """
        Sauvegarde les résultats au format JSON.
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'project_dir': self.root_dir,
            'modules': self.results,
            'dependencies': self.module_dependencies if self.include_dependencies else {}
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _save_markdown(self) -> None:
        """
        Sauvegarde les résultats au format Markdown.
        """
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Analyse du code Python\n\n")
            f.write(f"Projet: **{os.path.basename(self.root_dir)}**\n\n")
            f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table des matières
            f.write("## Table des matières\n\n")
            for file_path in sorted(self.results.keys()):
                file_anchor = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
                f.write(f"- [{file_path}](#{file_anchor})\n")
            f.write("\n")
            
            # Statistiques globales
            f.write("## Statistiques globales\n\n")
            
            total_files = len(self.results)
            total_functions = sum(len(data['functions']) for data in self.results.values())
            total_classes = sum(len(data['classes']) for data in self.results.values())
            total_methods = sum(sum(len(cls['methods']) for cls in data['classes']) for data in self.results.values())
            total_loc = sum(data['loc'] for data in self.results.values())
            
            f.write(f"- **Fichiers Python**: {total_files}\n")
            f.write(f"- **Fonctions**: {total_functions}\n")
            f.write(f"- **Classes**: {total_classes}\n")
            f.write(f"- **Méthodes**: {total_methods}\n")
            f.write(f"- **Lignes de code**: {total_loc}\n\n")
            
            # Graphique des dépendances (Mermaid)
            if self.include_dependencies and self.module_dependencies:
                f.write("## Graphique des dépendances\n\n")
                f.write("```mermaid\ngraph TD\n")
                
                for module, deps in self.module_dependencies.items():
                    module_id = module.replace('.', '_')
                    for dep in deps:
                        dep_id = dep.replace('.', '_')
                        f.write(f"  {module_id}[{module}] --> {dep_id}[{dep}]\n")
                
                f.write("```\n\n")
            
            # Détails par fichier
            for file_path, file_data in sorted(self.results.items()):
                file_anchor = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
                f.write(f"<a id='{file_anchor}'></a>\n")
                f.write(f"## {file_path}\n\n")
                
                f.write(f"Lignes de code: {file_data['loc']}\n\n")
                
                # Affichage des importations
                if self.include_imports and file_data['imports']:
                    f.write("### Importations\n\n")
                    for imp in sorted(file_data['imports'], key=lambda x: x['lineno']):
                        if imp['type'] == 'import':
                            if imp['alias']:
                                f.write(f"- `import {imp['module']} as {imp['alias']}` - ligne {imp['lineno']}\n")
                            else:
                                f.write(f"- `import {imp['module']}` - ligne {imp['lineno']}\n")
                        else:  # from import
                            module_parts = imp['module'].split('.')
                            from_part = '.'.join(module_parts[:-1]) if len(module_parts) > 1 else module_parts[0]
                            name_part = module_parts[-1] if len(module_parts) > 1 else '*'
                            
                            if imp['alias']:
                                f.write(f"- `from {from_part} import {name_part} as {imp['alias']}` - ligne {imp['lineno']}\n")
                            else:
                                f.write(f"- `from {from_part} import {name_part}` - ligne {imp['lineno']}\n")
                    f.write("\n")
                
                # Affichage des variables globales
                if file_data['global_vars']:
                    f.write("### Variables globales\n\n")
                    for var in sorted(file_data['global_vars'], key=lambda x: x['lineno']):
                        f.write(f"- `{var['name']}` - ligne {var['lineno']}\n")
                    f.write("\n")
                
                # Affichage des fonctions
                if file_data['functions']:
                    f.write("### Fonctions\n\n")
                    for func in sorted(file_data['functions'], key=lambda x: x['lineno']):
                        print(f"Trouvé la fonction {func['name']} dans le fichier {file_path}")
                        print(f"La fonction {func['name']} a les arguments suivants : {[arg['name'] for arg in func['args']]}")
                        args_str = ", ".join([self._format_arg(arg) for arg in func['args']])
                        
                        complexity_str = ""
                        if self.include_complexity and func['complexity'] is not None:
                            complexity_color = self._get_complexity_color(func['complexity'])
                            complexity_str = f" - <span style='color:{complexity_color}'>Complexité: {func['complexity']}</span>"
                        
                        return_str = ""
                        if func['return_type']:
                            return_str = f" -> {func['return_type']}"
                        
                        f.write(f"- `{func['name']}({args_str}){return_str}` - ligne {func['lineno']}{complexity_str}\n")
                        if func['docstring']:
                            f.write(f"  - *{func['docstring'].splitlines()[0]}*\n")
                    f.write("\n")
                
                # Affichage des classes
                if file_data['classes']:
                    f.write("### Classes\n\n")
                    for cls in sorted(file_data['classes'], key=lambda x: x['lineno']):
                        bases_str = ""
                        if cls['bases']:
                            bases_str = f"({', '.join(cls['bases'])})"
                        
                        f.write(f"- **{cls['name']}{bases_str}** - ligne {cls['lineno']}\n")
                        if cls['docstring']:
                            f.write(f"  - *{cls['docstring'].splitlines()[0]}*\n")
                        
                        if cls['class_vars']:
                            f.write("  - Variables de classe:\n")
                            for var in sorted(cls['class_vars'], key=lambda x: x['lineno']):
                                f.write(f"    - `{var['name']}` - ligne {var['lineno']}\n")
                        
                        if cls['methods']:
                            f.write("  - Méthodes:\n")
                            for method in sorted(cls['methods'], key=lambda x: x['lineno']):
                                args_str = ", ".join([self._format_arg(arg) for arg in method['args']])
                                
                                complexity_str = ""
                                if self.include_complexity and method['complexity'] is not None:
                                    complexity_color = self._get_complexity_color(method['complexity'])
                                    complexity_str = f" - <span style='color:{complexity_color}'>Complexité: {method['complexity']}</span>"
                                
                                return_str = ""
                                if method['return_type']:
                                    return_str = f" -> {method['return_type']}"
                                
                                f.write(f"    - `{method['name']}({args_str}){return_str}` - ligne {method['lineno']}{complexity_str}\n")
                                if method['docstring']:
                                    f.write(f"      - *{method['docstring'].splitlines()[0]}*\n")
                        f.write("\n")
                
                f.write("---\n\n")
    
    def _format_arg(self, arg: Dict[str, Any]) -> str:
        """
        Formate un argument de fonction pour l'affichage.
        """
        if arg['type']:
            return f"{arg['name']}: {arg['type']}"
        return arg['name']
    
    def _get_complexity_color(self, complexity: int) -> str:
        """
        Détermine la couleur en fonction de la complexité.
        """
        if complexity <= 5:
            return "green"
        elif complexity <= 10:
            return "orange"
        else:
            return "red"
    
    def _save_html(self) -> None:
        """
        Sauvegarde les résultats au format HTML avec navigation.
        """
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse de Code Python</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        h3 { color: #2980b9; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        code { font-family: Consolas, monospace; color: #e74c3c; }
        .complexity-low { color: green; }
        .complexity-medium { color: orange; }
        .complexity-high { color: red; }
        .toc { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .toc ul { list-style-type: none; padding-left: 20px; }
        .toc li { margin-bottom: 5px; }
        .toc a { text-decoration: none; color: #3498db; }
        .toc a:hover { text-decoration: underline; }
        .file-item { margin-bottom: 40px; }
        .stats { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
        .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; flex: 1; min-width: 150px; }
        .stat-number { font-size: 24px; font-weight: bold; color: #3498db; }
        .hidden { display: none; }
        .toggle-btn { background: #3498db; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; }
        .toggle-btn:hover { background: #2980b9; }
        .search-box { margin-bottom: 20px; padding: 10px; width: 100%; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
""")
            
            # Header
            f.write(f"<h1>Analyse du code Python</h1>\n")
            f.write(f"<p>Projet: <strong>{os.path.basename(self.root_dir)}</strong></p>\n")
            f.write(f"<p>Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Search box
            f.write('<input type="text" id="searchBox" class="search-box" placeholder="Rechercher des fonctions, classes...">\n')
            
            # Stats
            f.write("<h2>Statistiques globales</h2>\n")
            f.write("<div class='stats'>\n")
            
            total_files = len(self.results)
            total_functions = sum(len(data['functions']) for data in self.results.values())
            total_classes = sum(len(data['classes']) for data in self.results.values())
            total_methods = sum(sum(len(cls['methods']) for cls in data['classes']) for data in self.results.values())
            total_loc = sum(data['loc'] for data in self.results.values())
            
            f.write(f"<div class='stat-card'><div class='stat-number'>{total_files}</div>Fichiers Python</div>\n")
            f.write(f"<div class='stat-card'><div class='stat-number'>{total_functions}</div>Fonctions</div>\n")
            f.write(f"<div class='stat-card'><div class='stat-number'>{total_classes}</div>Classes</div>\n")
            f.write(f"<div class='stat-card'><div class='stat-number'>{total_methods}</div>Méthodes</div>\n")
            f.write(f"<div class='stat-card'><div class='stat-number'>{total_loc}</div>Lignes de code</div>\n")
            f.write("</div>\n")
            
            # Table of contents
            f.write("<div class='toc'>\n")
            f.write("<h2>Table des matières</h2>\n")
            f.write("<ul>\n")
            for file_path in sorted(self.results.keys()):
                file_id = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
                f.write(f"<li><a href='#{file_id}'>{file_path}</a></li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
            
            # File details
            for file_path, file_data in sorted(self.results.items()):
                file_id = file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
                f.write(f"<div id='{file_id}' class='file-item'>\n")
                f.write(f"<h2>{file_path}</h2>\n")
                f.write(f"<p>Lignes de code: {file_data['loc']}</p>\n")
                
                # Imports
                if self.include_imports and file_data['imports']:
                    f.write("<h3>Importations</h3>\n")
                    f.write("<button class='toggle-btn' onclick='toggleSection(\"imports-" + file_id + "\")'>Afficher/Masquer</button>\n")
                    f.write(f"<div id='imports-{file_id}'>\n")
                    f.write("<ul>\n")
                    for imp in sorted(file_data['imports'], key=lambda x: x['lineno']):
                        if imp['type'] == 'import':
                            if imp['alias']:
                                f.write(f"<li><code>import {imp['module']} as {imp['alias']}</code> - ligne {imp['lineno']}</li>\n")
                            else:
                                f.write(f"<li><code>import {imp['module']}</code> - ligne {imp['lineno']}</li>\n")
                        else:  # from import
                            module_parts = imp['module'].split('.')
                            from_part = '.'.join(module_parts[:-1]) if len(module_parts) > 1 else module_parts[0]
                            name_part = module_parts[-1] if len(module_parts) > 1 else '*'
                            
                            if imp['alias']:
                                f.write(f"<li><code>from {from_part} import {name_part} as {imp['alias']}</code> - ligne {imp['lineno']}</li>\n")
                            else:
                                f.write(f"<li><code>from {from_part} import {name_part}</code> - ligne {imp['lineno']}</li>\n")
                    f.write("</ul>\n")
                    f.write("</div>\n")
                
                # Global variables
                if file_data['global_vars']:
                    f.write("<h3>Variables globales</h3>\n")
                    f.write("<button class='toggle-btn' onclick='toggleSection(\"vars-" + file_id + "\")'>Afficher/Masquer</button>\n")
                    f.write(f"<div id='vars-{file_id}'>\n")
                    f.write("<ul>\n")
                    for var in sorted(file_data['global_vars'], key=lambda x: x['lineno']):
                        f.write(f"<li><code>{var['name']}</code> - ligne {var['lineno']}</li>\n")
                    f.write("</ul>\n")
                    f.write("</div>\n")
                
                # Functions
                if file_data['functions']:
                    f.write("<h3>Fonctions</h3>\n")
                    f.write("<button class='toggle-btn' onclick='toggleSection(\"functions-" + file_id + "\")'>Afficher/Masquer</button>\n")
                    f.write(f"<div id='functions-{file_id}'>\n")
                    f.write("<ul>\n")
                    for func in sorted(file_data['functions'], key=lambda x: x['lineno']):
                        args_str = ", ".join([self._format_arg(arg) for arg in func['args']])
                        
                        complexity_class = ""
                        complexity_str = ""
                        if self.include_complexity and func['complexity'] is not None:
                            if func['complexity'] <= 5:
                                complexity_class = "complexity-low"
                            elif func['complexity'] <= 10:
                                complexity_class = "complexity-medium"
                            else:
                                complexity_class = "complexity-high"
                            complexity_str = f" - <span class='{complexity_class}'>Complexité: {func['complexity']}</span>"
                        
                        return_str = ""
                        if func['return_type']:
                            return_str = f" -> {func['return_type']}"
                        
                        f.write(f"<li><code>{func['name']}({args_str}){return_str}</code> - ligne {func['lineno']}{complexity_str}</li>\n")
                        if func['docstring']:
                            f.write(f"<p><em>{func['docstring'].splitlines()[0]}</em></p>\n")
                    f.write("</ul>\n")
                    f.write("</div>\n")
                
                # Classes
                if file_data['classes']:
                    f.write("<h3>Classes</h3>\n")
                    f.write("<button class='toggle-btn' onclick='toggleSection(\"classes-" + file_id + "\")'>Afficher/Masquer</button>\n")
                    f.write(f"<div id='classes-{file_id}'>\n")
                    
                    for cls in sorted(file_data['classes'], key=lambda x: x['lineno']):
                        f.write(f"<div class='class-item'>\n")
                        f.write(f"<h4><code>{cls['name']}</code> - ligne {cls['lineno']}</h4>\n")
                        
                        if cls['docstring']:
                            f.write(f"<p><em>{cls['docstring'].splitlines()[0]}</em></p>\n")
                        
                        if cls['bases']:
                            f.write("<p>Hérite de: ")
                            f.write(", ".join([f"<code>{base}</code>" for base in cls['bases']]))
                            f.write("</p>\n")
                        
                        # Class variables
                        if cls['class_vars']:
                            f.write("<h5>Variables de classe:</h5>\n")
                            f.write("<ul>\n")
                            for var in sorted(cls['class_vars'], key=lambda x: x['lineno']):
                                f.write(f"<li><code>{var['name']}</code> - ligne {var['lineno']}</li>\n")
                            f.write("</ul>\n")
                        
                        # Methods
                        if cls['methods']:
                            f.write("<h5>Méthodes:</h5>\n")
                            f.write("<ul>\n")
                            for method in sorted(cls['methods'], key=lambda x: x['lineno']):
                                args_str = ", ".join([self._format_arg(arg) for arg in method['args']])
                                
                                complexity_class = ""
                                complexity_str = ""
                                if self.include_complexity and method['complexity'] is not None:
                                    if method['complexity'] <= 5:
                                        complexity_class = "complexity-low"
                                    elif method['complexity'] <= 10:
                                        complexity_class = "complexity-medium"
                                    else:
                                        complexity_class = "complexity-high"
                                    complexity_str = f" - <span class='{complexity_class}'>Complexité: {method['complexity']}</span>"
                                
                                return_str = ""
                                if method['return_type']:
                                    return_str = f" -> {method['return_type']}"
                                
                                f.write(f"<li><code>{method['name']}({args_str}){return_str}</code> - ligne {method['lineno']}{complexity_str}</li>\n")
                                if method['docstring']:
                                    f.write(f"<p><em>{method['docstring'].splitlines()[0]}</em></p>\n")
                            f.write("</ul>\n")
                        
                        f.write("</div>\n")
                    
                    f.write("</div>\n")
                
                f.write("</div>\n")
            
            # JavaScript for interactivity
            f.write("""
<script>
function toggleSection(id) {
    const element = document.getElementById(id);
    if (element.classList.contains('hidden')) {
        element.classList.remove('hidden');
    } else {
        element.classList.add('hidden');
    }
}

// Search functionality
document.getElementById('searchBox').addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    const fileItems = document.querySelectorAll('.file-item');
    
    fileItems.forEach(function(item) {
        if (searchTerm === '') {
            item.style.display = 'block';
            return;
        }
        
        const text = item.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
            item.style.display = 'block';
            
            // Ensure sections containing matches are visible
            const matchedSections = item.querySelectorAll('div[id^="functions-"], div[id^="classes-"], div[id^="imports-"], div[id^="vars-"]');
            matchedSections.forEach(function(section) {
                if (section.textContent.toLowerCase().includes(searchTerm)) {
                    section.classList.remove('hidden');
                }
            });
        } else {
            item.style.display = 'none';
        }
    });
});

// Initialize all toggleable sections to be hidden by default
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('[id^="functions-"], [id^="classes-"], [id^="imports-"], [id^="vars-"]');
    sections.forEach(function(section) {
        section.classList.add('hidden');
    });
});
</script>
""")
            
            f.write("""
    </div>
</body>
</html>
""")
    
    def generate_report(self):
        """
        Génère un rapport HTML à partir des données analysées.
        """
        if not self.results:
            print("Aucune donnée analysée. Le rapport sera vide.")
            return

        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("<html><body><h1>Rapport d'analyse</h1>\n")
            
            for file_path, file_data in self.results.items():
                f.write(f"<h2>{file_path}</h2>\n")
                
                if file_data['functions']:
                    f.write("<h3>Fonctions</h3>\n")
                    for func in file_data['functions']:
                        f.write(f"<p>{func['name']}({', '.join(arg['name'] for arg in func['args'])})</p>\n")
                
                if file_data['classes']:
                    f.write("<h3>Classes</h3>\n")
                    for cls in file_data['classes']:
                        f.write(f"<p>{cls['name']}</p>\n")
            
            f.write("</body></html>\n")
        
        print(f"Rapport généré dans {self.output_file}")

    def analyze_code(self, file_path):
        """
        Analyse les erreurs de structuration, les redondances, et la structure des définitions, fonctions, classes et imports dans le fichier.
        Cette méthode identifie également les définitions du même nom dans différents fichiers et les indentations incorrectes.

        Args:
            file_path (str): Chemin du fichier à analyser.

        Returns:
            None: Les résultats sont affichés directement dans la console.
        """
        from colorama import Fore, Style  # Pour l'utilisation des couleurs

        with open(file_path, "r") as file:
            code = file.read()

        # Parse le code en un AST
        tree = ast.parse(code)

        # Suivi des définitions de fonctions, classes, imports et décorateurs
        definitions = defaultdict(list)
        imports = []
        decorateurs = []

        # Vérification de l'indentation et de la structuration
        problemes_indentation = []
        erreurs_structuration = []

        # Analyse de l'AST
        for node in ast.walk(tree):
            # Suivi des imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                imports.append((f"from {node.module} import {', '.join([alias.name for alias in node.names])}", node.lineno))

            # Suivi des définitions de fonctions et classes
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name].append((node.lineno, node.col_offset))

                # Vérification des docstrings
                if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str)):
                    erreurs_structuration.append((node.lineno, "Docstring manquant ou incomplet"))

            # Suivi des décorateurs
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorateurs.append((decorator.id, node.lineno))

            # Vérification de l'indentation (plus rigoureuse)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)):
                # Vérifier l'indentation du nœud
                if not (node.col_offset % 4 == 0):
                    problemes_indentation.append((node.lineno, node.col_offset))

                # Vérifier l'indentation des enfants du nœud
                for child in ast.iter_child_nodes(node):
                    if hasattr(child, 'col_offset') and not (child.col_offset == node.col_offset + 4):
                        problemes_indentation.append((child.lineno, child.col_offset))

                # Vérifier les blocs mal structurés
                if not node.body:
                    erreurs_structuration.append((node.lineno, "Bloc vide ou mal structuré"))

        # Rapport des erreurs
        print(f"\n{Fore.RED}=== Erreurs ==={Style.RESET_ALL}")
        if erreurs_structuration:
            print(f"{Fore.RED}Erreurs de Structuration :{Style.RESET_ALL}")
            for ligne, message in erreurs_structuration:
                print(f"  {Fore.RED}Ligne {ligne} : {message}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Aucune erreur de structuration trouvée.{Style.RESET_ALL}")

        if problemes_indentation:
            print(f"{Fore.RED}Problèmes d'Indentation :{Style.RESET_ALL}")
            for ligne, col in problemes_indentation:
                print(f"  {Fore.RED}Ligne {ligne}, colonne {col} : Indentation incorrecte{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Aucun problème d'indentation trouvé.{Style.RESET_ALL}")

        # Rapport des redondances
        definitions_redondantes = {nom: lignes for nom, lignes in definitions.items() if len(lignes) > 1}
        if definitions_redondantes:
            print(f"\n{Fore.RED}=== Redondances ==={Style.RESET_ALL}")
            for nom, lignes in definitions_redondantes.items():
                print(f"  {Fore.RED}{nom}{Style.RESET_ALL} est défini plusieurs fois aux lignes : {[ligne[0] for ligne in lignes]}")
                print(f"  Fichier : {file_path}")
        else:
            print(f"\n{Fore.GREEN}Aucune redondance trouvée.{Style.RESET_ALL}")

        # Rapport de la structure du fichier
        print(f"\n{Fore.BLUE}=== Structure du Fichier ==={Style.RESET_ALL}")
        if imports:
            print(f"{Fore.BLUE}Imports :{Style.RESET_ALL}")
            for imp, ligne in imports:
                print(f"  {Fore.GREEN}Ligne {ligne} : {imp}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Aucun import trouvé.{Style.RESET_ALL}")

        if definitions:
            print(f"{Fore.BLUE}Définitions :{Style.RESET_ALL}")
            for nom, lignes in definitions.items():
                print(f"  {Fore.GREEN}{nom}{Style.RESET_ALL} est défini aux lignes : {[ligne[0] for ligne in lignes]}")
        else:
            print(f"{Fore.YELLOW}Aucune définition trouvée.{Style.RESET_ALL}")

        if decorateurs:
            print(f"{Fore.BLUE}Décorateurs :{Style.RESET_ALL}")
            for decorateur, ligne in decorateurs:
                print(f"  {Fore.YELLOW}@{decorateur}{Style.RESET_ALL} est utilisé à la ligne {ligne}")
        else:
            print(f"{Fore.YELLOW}Aucun décorateur trouvé.{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyseur de code Python")
    parser.add_argument("path", help="Chemin vers le fichier ou dossier à analyser")
    parser.add_argument("-o", "--output", default="code_analysis.html", help="Fichier de sortie (HTML)")
    parser.add_argument("--include-complexity", action="store_true", help="Inclure l'analyse de complexité cyclomatique")
    parser.add_argument("--include-imports", action="store_true", help="Inclure les importations dans l'analyse")
    parser.add_argument("--exclude", help="Motif d'exclusion (glob) pour ignorer certains fichiers/dossiers")
    
    args = parser.parse_args()
    
    analyzer = PythonCodeAnalyzer(
        root_dir=args.path,
        path=args.path, 
        output_file=args.output,
        include_complexity=args.include_complexity,
        include_imports=args.include_imports,
        exclude_pattern=args.exclude
    )
    
    analyzer.analyze()
    analyzer.generate_report()
    
    print(f"Analyse terminée. Rapport généré dans {args.output}")
