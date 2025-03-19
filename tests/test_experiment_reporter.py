import pytest
import numpy as np
from src.reporting.experiment_reporter import ExperimentReporter
import os
import tempfile
import shutil

@pytest.fixture
def reporter():
    """Fixture pour créer un reporter temporaire."""
    temp_dir = tempfile.mkdtemp()
    reporter = ExperimentReporter(save_dir=temp_dir)
    yield reporter
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data():
    """Fixture pour générer des données de test."""
    return {
        'experiment_name': 'test_experiment',
        'training_history': {
            'loss': [0.5, 0.4, 0.3],
            'val_loss': [0.55, 0.45, 0.35],
            'accuracy': [0.8, 0.85, 0.9]
        },
        'parameters': {
            'model': {
                'layers': [64, 32],
                'learning_rate': 0.001
            },
            'data': {
                'window_size': 60
            }
        },
        'performance_metrics': {
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15
        },
        'predictions': {
            'train': np.array([1.0, 1.1, 0.9]),
            'test': np.array([1.2, 0.8, 1.0])
        },
        'actual_values': np.array([1.1, 1.0, 0.95])
    }

def test_reporter_initialization(reporter):
    """Teste l'initialisation du reporter."""
    assert os.path.exists(reporter.save_dir)

def test_report_generation(reporter, sample_data):
    """Teste la génération complète d'un rapport."""
    report_path = reporter.generate_report(
        experiment_name=sample_data['experiment_name'],
        training_history=sample_data['training_history'],
        parameters=sample_data['parameters'],
        performance_metrics=sample_data['performance_metrics'],
        predictions=sample_data['predictions'],
        actual_values=sample_data['actual_values']
    )
    
    assert os.path.exists(report_path)
    assert report_path.endswith('.pdf')

def test_parameter_validation(reporter, sample_data):
    """Teste la validation des paramètres."""
    with pytest.raises(ValueError):
        reporter.generate_report(
            experiment_name="",  # Nom vide
            training_history=sample_data['training_history'],
            parameters=sample_data['parameters'],
            performance_metrics=sample_data['performance_metrics'],
            predictions=sample_data['predictions'],
            actual_values=sample_data['actual_values']
        )

def test_data_shape_validation(reporter, sample_data):
    """Teste la validation des formes des données."""
    invalid_predictions = {
        'train': np.array([1.0, 1.1]),  # Taille différente
        'test': np.array([1.2, 0.8, 1.0])
    }
    
    with pytest.raises(ValueError):
        reporter.generate_report(
            experiment_name=sample_data['experiment_name'],
            training_history=sample_data['training_history'],
            parameters=sample_data['parameters'],
            performance_metrics=sample_data['performance_metrics'],
            predictions=invalid_predictions,
            actual_values=sample_data['actual_values']
        )

def test_metric_validation(reporter, sample_data):
    """Teste la validation des métriques."""
    invalid_metrics = {
        'sharpe_ratio': 'invalid',  # Type invalide
        'max_drawdown': -0.15
    }
    
    with pytest.raises(TypeError):
        reporter.generate_report(
            experiment_name=sample_data['experiment_name'],
            training_history=sample_data['training_history'],
            parameters=sample_data['parameters'],
            performance_metrics=invalid_metrics,
            predictions=sample_data['predictions'],
            actual_values=sample_data['actual_values']
        )

def test_history_validation(reporter, sample_data):
    """Teste la validation de l'historique d'entraînement."""
    invalid_history = {
        'loss': [0.5, 0.4, None],  # Valeur invalide
        'val_loss': [0.55, 0.45, 0.35]
    }
    
    with pytest.raises(ValueError):
        reporter.generate_report(
            experiment_name=sample_data['experiment_name'],
            training_history=invalid_history,
            parameters=sample_data['parameters'],
            performance_metrics=sample_data['performance_metrics'],
            predictions=sample_data['predictions'],
            actual_values=sample_data['actual_values']
        )

def test_temporary_files_cleanup(reporter, sample_data):
    """Teste le nettoyage des fichiers temporaires."""
    report_path = reporter.generate_report(
        experiment_name=sample_data['experiment_name'],
        training_history=sample_data['training_history'],
        parameters=sample_data['parameters'],
        performance_metrics=sample_data['performance_metrics'],
        predictions=sample_data['predictions'],
        actual_values=sample_data['actual_values']
    )
    
    # Vérifie qu'aucun fichier temporaire ne reste
    temp_files = [f for f in os.listdir(reporter.save_dir) 
                 if f.startswith('temp_')]
    assert len(temp_files) == 0 