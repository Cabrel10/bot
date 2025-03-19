"""
Tests unitaires pour la validation du fichier de configuration.
"""

import unittest
import yaml
import os
from typing import Dict, Any
from datetime import datetime

from src.config.validation import (
    ConfigValidator,
    ConfigError,
    ValidationError,
    SchemaError
)

class TestConfigValidation(unittest.TestCase):
    """Tests pour la validation de la configuration."""
    
    def setUp(self):
        """Initialisation des données de test."""
        self.validator = ConfigValidator()
        
        # Configuration de base valide
        self.valid_config = {
            'exchange': {
                'name': 'binance',
                'market_type': 'spot',
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'timeframes': ['1h', '4h', '1d'],
                'start_date': '2024-01-01'
            },
            'trading': {
                'mode': 'backtest',
                'plugins': {
                    'indicators': {
                        'momentum': {
                            'enabled': True,
                            'rsi_window': 14,
                            'rsi_overbought': 70,
                            'rsi_oversold': 30,
                            'volume_ma_window': 20
                        },
                        'volatility': {
                            'enabled': True,
                            'atr_window': 14,
                            'bollinger_window': 20,
                            'bollinger_std': 2.0
                        }
                    },
                    'strategies': {
                        'ma_cross': {
                            'enabled': True,
                            'short_window': 10,
                            'long_window': 30,
                            'min_trend_strength': 0.02,
                            'volume_factor': 1.5
                        },
                        'rsi': {
                            'enabled': False,
                            'rsi_window': 14,
                            'overbought': 70,
                            'oversold': 30
                        }
                    }
                }
            },
            'backtest': {
                'initial_balance': 10000,
                'commission': 0.001,
                'slippage': 0.0005,
                'metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate']
            },
            'production': {
                'risk_management': {
                    'max_position_size': 1.0,
                    'max_leverage': 3.0,
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04
                },
                'monitoring': {
                    'enabled': True,
                    'metrics': ['pnl', 'exposure', 'margin_level'],
                    'alerts': ['margin_call', 'stop_loss_hit']
                }
            },
            'futures': {
                'enabled': True,
                'hedging': False,
                'margin_type': 'isolated',
                'funding_fee': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading.log',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'database': {
                'type': 'sqlite',
                'path': 'trading.db'
            },
            'cache': {
                'enabled': True,
                'type': 'redis',
                'host': 'localhost',
                'port': 6379
            },
            'security': {
                'api_key_encrypted': True,
                'ip_whitelist': ['127.0.0.1']
            }
        }
        
    def test_basic_validation(self):
        """Test de la validation de base de la configuration."""
        # Test de la configuration valide
        try:
            self.validator.validate(self.valid_config)
        except ConfigError:
            self.fail("La configuration valide a échoué la validation")
            
        # Test avec configuration vide
        with self.assertRaises(ValidationError):
            self.validator.validate({})
            
        # Test avec configuration None
        with self.assertRaises(ValidationError):
            self.validator.validate(None)
            
    def test_exchange_validation(self):
        """Test de la validation de la section exchange."""
        config = self.valid_config.copy()
        
        # Test avec exchange manquant
        config.pop('exchange')
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec symboles invalides
        config = self.valid_config.copy()
        config['exchange']['symbols'] = ['INVALID/PAIR']
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec timeframes invalides
        config = self.valid_config.copy()
        config['exchange']['timeframes'] = ['invalid']
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec date invalide
        config = self.valid_config.copy()
        config['exchange']['start_date'] = 'invalid_date'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_trading_validation(self):
        """Test de la validation de la section trading."""
        config = self.valid_config.copy()
        
        # Test avec mode invalide
        config['trading']['mode'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec plugin invalide
        config = self.valid_config.copy()
        config['trading']['plugins']['indicators']['invalid'] = {}
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec paramètres de stratégie invalides
        config = self.valid_config.copy()
        config['trading']['plugins']['strategies']['ma_cross']['short_window'] = -1
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_backtest_validation(self):
        """Test de la validation de la section backtest."""
        config = self.valid_config.copy()
        
        # Test avec balance initiale négative
        config['backtest']['initial_balance'] = -1000
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec commission invalide
        config = self.valid_config.copy()
        config['backtest']['commission'] = -0.1
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec métrique invalide
        config = self.valid_config.copy()
        config['backtest']['metrics'].append('invalid_metric')
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_production_validation(self):
        """Test de la validation de la section production."""
        config = self.valid_config.copy()
        
        # Test avec taille de position invalide
        config['production']['risk_management']['max_position_size'] = 2.0
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec levier invalide
        config = self.valid_config.copy()
        config['production']['risk_management']['max_leverage'] = -1.0
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec alerte invalide
        config = self.valid_config.copy()
        config['production']['monitoring']['alerts'].append('invalid_alert')
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_futures_validation(self):
        """Test de la validation de la section futures."""
        config = self.valid_config.copy()
        
        # Test avec type de marge invalide
        config['futures']['margin_type'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec hedging invalide
        config = self.valid_config.copy()
        config['futures']['hedging'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_logging_validation(self):
        """Test de la validation de la section logging."""
        config = self.valid_config.copy()
        
        # Test avec niveau de log invalide
        config['logging']['level'] = 'INVALID'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec format invalide
        config = self.valid_config.copy()
        config['logging']['format'] = 123
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_database_validation(self):
        """Test de la validation de la section database."""
        config = self.valid_config.copy()
        
        # Test avec type de base de données invalide
        config['database']['type'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec chemin invalide
        config = self.valid_config.copy()
        config['database']['path'] = 123
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_cache_validation(self):
        """Test de la validation de la section cache."""
        config = self.valid_config.copy()
        
        # Test avec type de cache invalide
        config['cache']['type'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec port invalide
        config = self.valid_config.copy()
        config['cache']['port'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_security_validation(self):
        """Test de la validation de la section security."""
        config = self.valid_config.copy()
        
        # Test avec api_key_encrypted invalide
        config['security']['api_key_encrypted'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test avec ip_whitelist invalide
        config = self.valid_config.copy()
        config['security']['ip_whitelist'] = 'invalid'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
    def test_schema_validation(self):
        """Test de la validation du schéma."""
        # Test avec schéma invalide
        with self.assertRaises(SchemaError):
            ConfigValidator(schema_path='invalid_path.yaml')
            
        # Test avec schéma mal formé
        with self.assertRaises(SchemaError):
            ConfigValidator(schema={"invalid": "schema"})
            
    def test_file_validation(self):
        """Test de la validation à partir d'un fichier."""
        # Création d'un fichier de test
        test_file = 'test_config.yaml'
        with open(test_file, 'w') as f:
            yaml.dump(self.valid_config, f)
            
        try:
            # Test de la validation du fichier
            try:
                self.validator.validate_file(test_file)
            except ConfigError:
                self.fail("La validation du fichier a échoué")
                
            # Test avec fichier inexistant
            with self.assertRaises(ConfigError):
                self.validator.validate_file('nonexistent.yaml')
                
            # Test avec fichier mal formé
            with open(test_file, 'w') as f:
                f.write('invalid: yaml: content')
            with self.assertRaises(ConfigError):
                self.validator.validate_file(test_file)
                
        finally:
            # Nettoyage
            if os.path.exists(test_file):
                os.remove(test_file)
                
    def test_type_validation(self):
        """Test de la validation des types."""
        config = self.valid_config.copy()
        
        # Test avec types invalides
        invalid_types = {
            'exchange.symbols': 'string',
            'backtest.initial_balance': 'string',
            'production.risk_management.max_leverage': 'string',
            'futures.hedging': 'string',
            'logging.level': 123,
            'database.path': 123,
            'cache.port': 'string',
            'security.ip_whitelist': 'string'
        }
        
        for path, value in invalid_types.items():
            temp_config = config.copy()
            parts = path.split('.')
            target = temp_config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            
            with self.assertRaises(ValidationError):
                self.validator.validate(temp_config)
                
    def test_dependency_validation(self):
        """Test de la validation des dépendances."""
        config = self.valid_config.copy()
        
        # Test des dépendances futures
        config['futures']['enabled'] = True
        config['exchange']['market_type'] = 'spot'
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test des dépendances de monitoring
        config = self.valid_config.copy()
        config['production']['monitoring']['enabled'] = True
        config['production']['monitoring']['metrics'] = []
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
        # Test des dépendances de cache
        config = self.valid_config.copy()
        config['cache']['enabled'] = True
        config['cache'].pop('type')
        with self.assertRaises(ValidationError):
            self.validator.validate(config)
            
if __name__ == '__main__':
    unittest.main() 