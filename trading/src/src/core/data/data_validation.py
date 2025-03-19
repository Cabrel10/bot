"""
Module de validation des données avec support avancé pour les futures et la validation sophistiquée.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass

from trading.utils.logging.logger import TradingLogger
from trading.core.exceptions import ValidationError

class ValidationLevel(Enum):
    """Niveaux de validation disponibles."""
    BASIC = "basic"  # Validation basique des paramètres
    INTERMEDIATE = "intermediate"  # Inclut la validation des données temporelles
    ADVANCED = "advanced"  # Inclut les vérifications de risque
    FULL = "full"  # Inclut toutes les validations

@dataclass
class ValidationConfig:
    """Configuration de validation."""
    level: ValidationLevel = ValidationLevel.FULL
    min_data_points: int = 1000
    max_missing_values: float = 0.01
    correlation_threshold: float = 0.95
    price_threshold: float = 0.1  # 10% variation max
    volume_threshold: float = 1000000
    gap_threshold: int = 300  # 5 minutes en secondes
    futures_config: Optional[Dict] = None
    custom_validators: Optional[List[callable]] = None

@dataclass
class ValidationResult:
    """Résultat détaillé de la validation."""
    is_valid: bool
    timestamp: datetime
    data_type: str
    missing_values: Dict[str, float]
    outliers: Dict[str, List[int]]
    data_quality_score: float
    errors: List[str]
    warnings: List[str]
    metadata: Optional[Dict[str, Any]] = None

class DataValidator:
    """Validation sophistiquée des données pour le trading."""
    
    def __init__(self, config: Optional[Union[Dict, ValidationConfig]] = None):
        """
        Initialise le validateur avec une configuration flexible.
        
        Args:
            config: Configuration de validation (dict ou ValidationConfig)
        """
        self.logger = TradingLogger()
        
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        elif isinstance(config, ValidationConfig):
            self.config = config
        else:
            self.config = ValidationConfig()
            
        self.validation_history = []
        
    def _create_config_from_dict(self, config_dict: Dict) -> ValidationConfig:
        """Crée une configuration à partir d'un dictionnaire."""
        level = ValidationLevel[config_dict.get('level', 'FULL').upper()]
        return ValidationConfig(
            level=level,
            min_data_points=config_dict.get('min_data_points', 1000),
            max_missing_values=config_dict.get('max_missing_values', 0.01),
            correlation_threshold=config_dict.get('correlation_threshold', 0.95),
            price_threshold=config_dict.get('price_threshold', 0.1),
            volume_threshold=config_dict.get('volume_threshold', 1000000),
            gap_threshold=config_dict.get('gap_threshold', 300),
            futures_config=config_dict.get('futures_config'),
            custom_validators=config_dict.get('custom_validators')
        )
        
    def validate_market_data(self, data: pd.DataFrame, data_type: str = 'spot') -> ValidationResult:
        """
        Valide les données de marché avec gestion sophistiquée des erreurs.
        
        Args:
            data: DataFrame à valider
            data_type: Type de données ('spot', 'futures', 'options')
            
        Returns:
            ValidationResult: Résultat détaillé de la validation
            
        Raises:
            ValidationError: Si les données sont invalides
        """
        start_time = datetime.now()
        result = ValidationResult(
            is_valid=True,
            timestamp=start_time,
            data_type=data_type,
            missing_values={},
            outliers={},
            data_quality_score=1.0,
            errors=[],
            warnings=[]
        )
        
        try:
            # Validation de base
            self._validate_basic(data, result)
            if not result.is_valid:
                return result
                
            # Validation intermédiaire
            if self.config.level.value >= ValidationLevel.INTERMEDIATE.value:
                self._validate_temporal(data, result)
                
            # Validation avancée
            if self.config.level.value >= ValidationLevel.ADVANCED.value:
                self._validate_advanced(data, result)
                
            # Validation complète
            if self.config.level.value >= ValidationLevel.FULL.value:
                self._validate_full(data, result, data_type)
                
            # Validation personnalisée
            if self.config.custom_validators:
                self._run_custom_validators(data, result)
                
            # Calcul du score de qualité
            result.data_quality_score = self._calculate_quality_score(result)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Erreur inattendue: {str(e)}")
            self.logger.error(f"Erreur lors de la validation: {str(e)}")
            
        finally:
            # Ajout des métadonnées
            result.metadata = {
                'validation_time': (datetime.now() - start_time).total_seconds(),
                'config_level': self.config.level.value,
                'rows_count': len(data),
                'columns_count': len(data.columns)
            }
            
            # Enregistrement de l'historique
            self.validation_history.append(result)
            
        return result
        
    def _validate_basic(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validation basique des données."""
        # Vérification des colonnes requises
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            result.is_valid = False
            result.errors.append(f"Colonnes manquantes: {missing_cols}")
            return
            
        # Vérification du nombre minimum de points
        if len(data) < self.config.min_data_points:
            result.is_valid = False
            result.errors.append(
                f"Nombre insuffisant de points de données: {len(data)} < {self.config.min_data_points}"
            )
            return
            
        # Vérification des valeurs manquantes
        missing_values = data[list(required_cols)].isnull().mean()
        result.missing_values = missing_values.to_dict()
        if (missing_values > self.config.max_missing_values).any():
            result.is_valid = False
            result.errors.append("Trop de valeurs manquantes")
            return
            
        # Vérification des types de données
        for col in required_cols:
            if not np.issubdtype(data[col].dtype, np.number):
                result.is_valid = False
                result.errors.append(f"Colonne {col} n'est pas numérique")
                return
                
    def _validate_temporal(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validation des aspects temporels."""
        try:
            # Vérification de l'ordre chronologique
            if not data.index.is_monotonic_increasing:
                result.warnings.append("Les données ne sont pas dans l'ordre chronologique")
                
            # Vérification des gaps temporels
            if hasattr(data.index, 'to_series'):
                gaps = data.index.to_series().diff().dt.total_seconds()
                large_gaps = gaps[gaps > self.config.gap_threshold]
                if not large_gaps.empty:
                    result.warnings.append(
                        f"Gaps temporels détectés: {len(large_gaps)} gaps > {self.config.gap_threshold}s"
                    )
                    
        except Exception as e:
            result.warnings.append(f"Erreur lors de la validation temporelle: {str(e)}")
            
    def _validate_advanced(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validation avancée incluant la qualité des données."""
        try:
            # Détection des outliers
            for col in ['open', 'high', 'low', 'close', 'volume']:
                outliers = self._detect_outliers(data[col])
                if len(outliers) > 0:
                    result.outliers[col] = outliers
                    result.warnings.append(f"Outliers détectés dans {col}: {len(outliers)} points")
                    
            # Vérification des variations de prix
            price_changes = data['close'].pct_change().abs()
            suspicious_changes = price_changes[price_changes > self.config.price_threshold]
            if not suspicious_changes.empty:
                result.warnings.append(
                    f"Variations de prix suspectes: {len(suspicious_changes)} variations > {self.config.price_threshold*100}%"
                )
                
            # Vérification des volumes
            suspicious_volumes = data['volume'][data['volume'] > self.config.volume_threshold]
            if not suspicious_volumes.empty:
                result.warnings.append(
                    f"Volumes suspects: {len(suspicious_volumes)} volumes > {self.config.volume_threshold}"
                )
                
        except Exception as e:
            result.warnings.append(f"Erreur lors de la validation avancée: {str(e)}")
            
    def _validate_full(self, data: pd.DataFrame, result: ValidationResult, data_type: str) -> None:
        """Validation complète incluant les spécificités du type de données."""
        try:
            # Validation spécifique aux futures
            if data_type == 'futures':
                self._validate_futures_data(data, result)
                
            # Vérification des corrélations
            correlation_matrix = data.corr().abs()
            mask = np.triu(np.ones_like(correlation_matrix), k=1)
            high_corr = correlation_matrix * mask > self.config.correlation_threshold
            if high_corr.any().any():
                pairs = np.where(high_corr)
                corr_pairs = list(zip(correlation_matrix.index[pairs[0]], 
                                    correlation_matrix.columns[pairs[1]]))
                result.warnings.append(
                    f"Corrélations élevées détectées: {len(corr_pairs)} paires > {self.config.correlation_threshold}"
                )
                
            # Vérification de la cohérence des prix OHLC
            if not self._validate_ohlc_consistency(data):
                result.is_valid = False
                result.errors.append("Incohérence dans les prix OHLC")
                
        except Exception as e:
            result.warnings.append(f"Erreur lors de la validation complète: {str(e)}")
            
    def _validate_futures_data(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Validation spécifique aux données futures."""
        if not self.config.futures_config:
            result.warnings.append("Configuration futures manquante")
            return
            
        try:
            # Vérification des colonnes spécifiques aux futures
            futures_cols = {'funding_rate', 'open_interest', 'next_funding_time'}
            missing_futures_cols = futures_cols - set(data.columns)
            if missing_futures_cols:
                result.warnings.append(f"Colonnes futures manquantes: {missing_futures_cols}")
                
            if 'funding_rate' in data.columns:
                # Vérification des funding rates
                if (data['funding_rate'].abs() > 0.01).any():  # Max 1%
                    result.warnings.append("Funding rates suspects détectés")
                    
            if 'open_interest' in data.columns:
                # Vérification de l'open interest
                if (data['open_interest'] < 0).any():
                    result.is_valid = False
                    result.errors.append("Open interest négatif détecté")
                    
        except Exception as e:
            result.warnings.append(f"Erreur lors de la validation futures: {str(e)}")
            
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """Détecte les outliers dans une série."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> bool:
        """Vérifie la cohérence des prix OHLC."""
        return (
            (data['low'] <= data['high']).all() and
            (data['low'] <= data['open']).all() and
            (data['low'] <= data['close']).all() and
            (data['high'] >= data['open']).all() and
            (data['high'] >= data['close']).all()
        )
        
    def _run_custom_validators(self, data: pd.DataFrame, result: ValidationResult) -> None:
        """Exécute les validateurs personnalisés."""
        if not self.config.custom_validators:
            return
            
        for validator in self.config.custom_validators:
            try:
                validator_result = validator(data)
                if isinstance(validator_result, tuple):
                    is_valid, message = validator_result
                    if not is_valid:
                        result.warnings.append(f"Validation personnalisée: {message}")
                elif not validator_result:
                    result.warnings.append("Échec de la validation personnalisée")
            except Exception as e:
                result.warnings.append(f"Erreur dans le validateur personnalisé: {str(e)}")
                
    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calcule un score de qualité des données."""
        score = 1.0
        
        # Pénalités pour les erreurs
        score -= len(result.errors) * 0.2
        
        # Pénalités pour les avertissements
        score -= len(result.warnings) * 0.1
        
        # Pénalités pour les valeurs manquantes
        if result.missing_values:
            avg_missing = np.mean(list(result.missing_values.values()))
            score -= avg_missing
            
        # Pénalités pour les outliers
        if result.outliers:
            total_outliers = sum(len(outliers) for outliers in result.outliers.values())
            score -= total_outliers * 0.01
            
        return max(0.0, min(1.0, score))  # Normalisation entre 0 et 1

    def get_validation_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des validations effectuées."""
        if not self.validation_history:
            return {"message": "Aucune validation effectuée"}
            
        return {
            "total_validations": len(self.validation_history),
            "success_rate": sum(1 for r in self.validation_history if r.is_valid) / len(self.validation_history),
            "average_quality_score": np.mean([r.data_quality_score for r in self.validation_history]),
            "common_errors": self._get_common_issues([r.errors for r in self.validation_history]),
            "common_warnings": self._get_common_issues([r.warnings for r in self.validation_history]),
            "last_validation": self.validation_history[-1].__dict__
        }
        
    def _get_common_issues(self, issues_list: List[List[str]], top_n: int = 5) -> List[Tuple[str, int]]:
        """Identifie les problèmes les plus courants."""
        from collections import Counter
        all_issues = [issue for sublist in issues_list for issue in sublist]
        return Counter(all_issues).most_common(top_n) 