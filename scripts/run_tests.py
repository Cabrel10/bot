#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def run_unit_tests(verbose=False):
    """Lance les tests unitaires"""
    cmd = ["pytest", "tests/unit", "-v"] if verbose else ["pytest", "tests/unit"]
    return subprocess.run(cmd).returncode == 0

def run_integration_tests(verbose=False):
    """Lance les tests d'intégration"""
    cmd = ["pytest", "tests/integration", "-v"] if verbose else ["pytest", "tests/integration"]
    return subprocess.run(cmd).returncode == 0

def run_model_tests(verbose=False):
    """Lance les tests spécifiques aux modèles"""
    cmd = ["pytest", "tests/models", "-v"] if verbose else ["pytest", "tests/models"]
    return subprocess.run(cmd).returncode == 0

def run_linting():
    """Lance les vérifications de style de code"""
    print("Running black...")
    black_result = subprocess.run(["black", "--check", "src"]).returncode == 0
    
    print("Running flake8...")
    flake8_result = subprocess.run(["flake8", "src"]).returncode == 0
    
    print("Running mypy...")
    mypy_result = subprocess.run(["mypy", "src"]).returncode == 0
    
    return all([black_result, flake8_result, mypy_result])

def main():
    parser = argparse.ArgumentParser(description="Script de test automatisé")
    parser.add_argument("--unit", action="store_true", help="Lance les tests unitaires")
    parser.add_argument("--integration", action="store_true", help="Lance les tests d'intégration")
    parser.add_argument("--model", action="store_true", help="Lance les tests des modèles")
    parser.add_argument("--lint", action="store_true", help="Lance les vérifications de style")
    parser.add_argument("--all", action="store_true", help="Lance tous les tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    
    args = parser.parse_args()
    
    if not any([args.unit, args.integration, args.model, args.lint, args.all]):
        parser.print_help()
        sys.exit(1)
    
    success = True
    
    if args.all or args.unit:
        print("Running unit tests...")
        success &= run_unit_tests(args.verbose)
        
    if args.all or args.integration:
        print("Running integration tests...")
        success &= run_integration_tests(args.verbose)
        
    if args.all or args.model:
        print("Running model tests...")
        success &= run_model_tests(args.verbose)
        
    if args.all or args.lint:
        print("Running linting...")
        success &= run_linting()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 