"""
Module de nettoyage et validation des données de marché.
"""
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from trading.core.data_types import MarketData
from trading.utils.logging import TradingLogger

@dataclass
class CleaningConfig:
    """Configuration pour le nettoyage des données."""
    remove_outliers: bool = True
    fill_missing: bool = True
    outlier_std_threshold: float = 3.0
    max_missing_ratio: float = 0.1
    interpolation_method: str = 'linear'
    min_price_value: float = 0.0
    max_volume_zeros: int = 5
    check_timestamp_continuity: bool = True
    rolling_window: int = 20

class DataCleaner:
    """Gestionnaire de nettoyage et validation des données."""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.logger = TradingLogger()
        self._anomaly_stats = {}
        self._cleaning_history = []

    def clean_market_data(self, data: MarketData) -> Tuple[MarketData, Dict]:
        """
        Nettoie et valide les données de marché.
        
        Args:
            data: Données de marché brutes
            
        Returns:
            Tuple[MarketData, Dict]: Données nettoyées et rapport de nettoyage
        """
        cleaning_report = {
            'original_shape': len(data.timestamp),
            'anomalies_detected': {},
            'corrections_made': {},
            'data_quality_score': 0.0
        }
        
        try:
            # Conversion en DataFrame pour faciliter le traitement
            df = self._to_dataframe(data)
            
            # Vérification de la continuité temporelle
            if self.config.check_timestamp_continuity:
                df = self._check_timestamp_continuity(df, cleaning_report)
            
            # Détection et traitement des valeurs aberrantes
            if self.config.remove_outliers:
                df = self._handle_outliers(df, cleaning_report)
            
            # Traitement des valeurs manquantes
            if self.config.fill_missing:
                df = self._handle_missing_values(df, cleaning_report)
            
            # Validation des prix et volumes
            df = self._validate_price_volume(df, cleaning_report)
            
            # Calcul du score de qualité
            cleaning_report['data_quality_score'] = self._calculate_quality_score(df, cleaning_report)
            
            # Conversion retour en MarketData
            cleaned_data = self._to_market_data(df)
            
            # Mise à jour des statistiques
            self._update_cleaning_stats(cleaning_report)
            
            return cleaned_data, cleaning_report
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des données: {e}")
            raise

    def _to_dataframe(self, data: MarketData) -> pd.DataFrame:
        """Convertit MarketData en DataFrame."""
        df = pd.DataFrame({
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        }, index=data.timestamp)
        
        if data.indicators is not None:
            for col in data.indicators.columns:
                df[col] = data.indicators[col]
        
        return df

    def _check_timestamp_continuity(self, 
                                  df: pd.DataFrame,
                                  report: Dict) -> pd.DataFrame:
        """Vérifie et corrige la continuité temporelle."""
        # Vérification des intervalles
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=1)  # À ajuster selon le timeframe
        
        gaps = time_diff[time_diff > expected_diff]
        report['anomalies_detected']['timestamp_gaps'] = len(gaps)
        
        if len(gaps) > 0:
            # Création d'un index continu
            full_index = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq='1min'  # À ajuster selon le timeframe
            )
            
            # Réindexation avec remplissage
            df = df.reindex(full_index)
            report['corrections_made']['filled_timestamp_gaps'] = len(gaps)
        
        return df

    def _handle_outliers(self, 
                        df: pd.DataFrame,
                        report: Dict) -> pd.DataFrame:
        """Détecte et traite les valeurs aberrantes."""
        outliers_detected = {}
        
        for column in ['open', 'high', 'low', 'close', 'volume']:
            # Calcul des z-scores
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = z_scores > self.config.outlier_std_threshold
            
            outliers_detected[column] = sum(outliers)
            
            if sum(outliers) > 0:
                # Remplacement des valeurs aberrantes par la moyenne mobile
                rolling_mean = df[column].rolling(
                    window=self.config.rolling_window,
                    center=True
                ).mean()
                
                df.loc[outliers, column] = rolling_mean[outliers]
        
        report['anomalies_detected']['outliers'] = outliers_detected
        report['corrections_made']['outliers_corrected'] = sum(outliers_detected.values())
        
        return df

    def _handle_missing_values(self, 
                             df: pd.DataFrame,
                             report: Dict) -> pd.DataFrame:
        """Traite les valeurs manquantes."""
        missing_counts = df.isnull().sum()
        report['anomalies_detected']['missing_values'] = missing_counts.to_dict()
        
        # Vérification du ratio de valeurs manquantes
        missing_ratio = missing_counts / len(df)
        if (missing_ratio > self.config.max_missing_ratio).any():
            problematic_cols = missing_ratio[missing_ratio > self.config.max_missing_ratio].index
            self.logger.warning(
                f"Colonnes avec trop de valeurs manquantes: {problematic_cols}"
            )
        
        # Interpolation des valeurs manquantes
        df = df.interpolate(method=self.config.interpolation_method)
        
        # Remplissage des valeurs restantes
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        report['corrections_made']['missing_values_filled'] = missing_counts.to_dict()
        
        return df

    def _validate_price_volume(self, 
                             df: pd.DataFrame,
                             report: Dict) -> pd.DataFrame:
        """Valide les prix et volumes."""
        # Validation des prix
        invalid_prices = (df[['open', 'high', 'low', 'close']] < self.config.min_price_value).any(axis=1)
        report['anomalies_detected']['invalid_prices'] = sum(invalid_prices)
        
        # Validation des volumes
        zero_volumes = df['volume'] == 0
        consecutive_zeros = self._find_consecutive_zeros(zero_volumes)
        report['anomalies_detected']['suspicious_volume_patterns'] = consecutive_zeros
        
        if consecutive_zeros > self.config.max_volume_zeros:
            self.logger.warning(f"Séquences suspectes de volumes nuls détectées: {consecutive_zeros}")
        
        return df

    def _find_consecutive_zeros(self, zero_series: pd.Series) -> int:
        """Trouve les séquences de zéros consécutifs."""
        consecutive_count = 0
        max_consecutive = 0
        
        for value in zero_series:
            if value:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_consecutive

    def _calculate_quality_score(self, 
                               df: pd.DataFrame,
                               report: Dict) -> float:
        """Calcule un score de qualité pour les données."""
        # Facteurs de qualité
        completeness = 1 - df.isnull().mean().mean()
        outlier_ratio = sum(report['corrections_made'].get('outliers_corrected', 0)) / len(df)
        timestamp_quality = 1 - (report['anomalies_detected'].get('timestamp_gaps', 0) / len(df))
        
        # Pondération des facteurs
        weights = {
            'completeness': 0.4,
            'outlier_ratio': 0.3,
            'timestamp_quality': 0.3
        }
        
        quality_score = (
            weights['completeness'] * completeness +
            weights['outlier_ratio'] * (1 - outlier_ratio) +
            weights['timestamp_quality'] * timestamp_quality
        )
        
        return quality_score

    def _to_market_data(self, df: pd.DataFrame) -> MarketData:
        """Convertit DataFrame en MarketData."""
        indicator_columns = [col for col in df.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return MarketData(
            timestamp=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            indicators=df[indicator_columns] if indicator_columns else None
        )

    def _update_cleaning_stats(self, report: Dict):
        """Met à jour les statistiques de nettoyage."""
        self._cleaning_history.append({
            'timestamp': datetime.now(),
            'report': report
        })
        
        # Mise à jour des statistiques globales
        for anomaly_type, count in report['anomalies_detected'].items():
            if anomaly_type not in self._anomaly_stats:
                self._anomaly_stats[anomaly_type] = []
            self._anomaly_stats[anomaly_type].append(count)

    def get_cleaning_stats(self) -> Dict:
        """Retourne les statistiques de nettoyage."""
        stats = {
            'total_cleanings': len(self._cleaning_history),
            'average_quality_score': np.mean([
                report['report']['data_quality_score']
                for report in self._cleaning_history
            ]),
            'anomaly_trends': {
                anomaly_type: {
                    'mean': np.mean(counts),
                    'std': np.std(counts),
                    'trend': np.polyfit(range(len(counts)), counts, 1)[0]
                }
                for anomaly_type, counts in self._anomaly_stats.items()
            }
        }
        
        return stats 