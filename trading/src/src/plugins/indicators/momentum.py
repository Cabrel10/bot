"""
Module d'indicateurs de momentum pour le système de trading.
Implémente les indicateurs techniques RSI, MACD et ADX.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class RSIConfig:
    """Configuration pour l'indicateur RSI."""
    period: int = 14
    overbought: float = 70
    oversold: float = 30

@dataclass
class MACDConfig:
    """Configuration pour l'indicateur MACD."""
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

@dataclass
class ADXConfig:
    """Configuration pour l'indicateur ADX."""
    period: int = 14
    threshold: float = 25

class RSI:
    """Relative Strength Index (RSI)."""
    
    def __init__(self, config: RSIConfig = RSIConfig()):
        """
        Initialise l'indicateur RSI.
        
        Args:
            config: Configuration de l'indicateur RSI
        """
        self.config = config
        self.values: List[float] = []
        self.gains: List[float] = []
        self.losses: List[float] = []
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        
    def calculate(self, prices: List[float]) -> float:
        """
        Calcule la valeur du RSI pour une série de prix.
        
        Args:
            prices: Liste des prix de clôture
            
        Returns:
            float: Valeur du RSI
        """
        if len(prices) < self.config.period + 1:
            return 50.0  # Valeur neutre par défaut
            
        # Calcul des variations
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, change) for change in changes]
        losses = [abs(min(0, change)) for change in changes]
        
        # Calcul des moyennes
        if self.avg_gain is None or self.avg_loss is None:
            self.avg_gain = sum(gains[:self.config.period]) / self.config.period
            self.avg_loss = sum(losses[:self.config.period]) / self.config.period
        else:
            self.avg_gain = (self.avg_gain * (self.config.period - 1) + gains[-1]) / self.config.period
            self.avg_loss = (self.avg_loss * (self.config.period - 1) + losses[-1]) / self.config.period
        
        # Calcul du RSI
        if self.avg_loss == 0:
            return 100.0
        
        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.values.append(rsi)
        return rsi
    
    def is_overbought(self) -> bool:
        """Vérifie si le marché est suracheté."""
        return len(self.values) > 0 and self.values[-1] >= self.config.overbought
    
    def is_oversold(self) -> bool:
        """Vérifie si le marché est survendu."""
        return len(self.values) > 0 and self.values[-1] <= self.config.oversold

class MACD:
    """Moving Average Convergence Divergence (MACD)."""
    
    def __init__(self, config: MACDConfig = MACDConfig()):
        """
        Initialise l'indicateur MACD.
        
        Args:
            config: Configuration de l'indicateur MACD
        """
        self.config = config
        self.values: List[Dict[str, float]] = []
        self.fast_ema: List[float] = []
        self.slow_ema: List[float] = []
        self.signal_line: List[float] = []
        
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calcule l'EMA pour une période donnée.
        
        Args:
            prices: Liste des prix
            period: Période de l'EMA
            
        Returns:
            float: Valeur de l'EMA
        """
        if len(prices) < period:
            return prices[-1]
            
        alpha = 2 / (period + 1)
        if len(self.fast_ema if period == self.config.fast_period else self.slow_ema) == 0:
            ema = sum(prices[-period:]) / period
        else:
            prev_ema = self.fast_ema[-1] if period == self.config.fast_period else self.slow_ema[-1]
            ema = (prices[-1] - prev_ema) * alpha + prev_ema
            
        return ema
    
    def calculate(self, prices: List[float]) -> Dict[str, float]:
        """
        Calcule les valeurs du MACD.
        
        Args:
            prices: Liste des prix de clôture
            
        Returns:
            Dict[str, float]: Valeurs du MACD (macd_line, signal_line, histogram)
        """
        # Calcul des EMAs
        fast_ema = self.calculate_ema(prices, self.config.fast_period)
        slow_ema = self.calculate_ema(prices, self.config.slow_period)
        
        self.fast_ema.append(fast_ema)
        self.slow_ema.append(slow_ema)
        
        # Calcul de la ligne MACD
        macd_line = fast_ema - slow_ema
        
        # Calcul de la ligne de signal
        if len(self.signal_line) == 0:
            signal = macd_line
        else:
            alpha = 2 / (self.config.signal_period + 1)
            signal = (macd_line - self.signal_line[-1]) * alpha + self.signal_line[-1]
        
        self.signal_line.append(signal)
        
        # Calcul de l'histogramme
        histogram = macd_line - signal
        
        result = {
            'macd_line': macd_line,
            'signal_line': signal,
            'histogram': histogram
        }
        
        self.values.append(result)
        return result
    
    def is_bullish_crossover(self) -> bool:
        """Vérifie s'il y a un croisement haussier."""
        if len(self.values) < 2:
            return False
        return (self.values[-2]['macd_line'] <= self.values[-2]['signal_line'] and
                self.values[-1]['macd_line'] > self.values[-1]['signal_line'])
    
    def is_bearish_crossover(self) -> bool:
        """Vérifie s'il y a un croisement baissier."""
        if len(self.values) < 2:
            return False
        return (self.values[-2]['macd_line'] >= self.values[-2]['signal_line'] and
                self.values[-1]['macd_line'] < self.values[-1]['signal_line'])

class ADX:
    """Average Directional Index (ADX)."""
    
    def __init__(self, config: ADXConfig = ADXConfig()):
        """
        Initialise l'indicateur ADX.
        
        Args:
            config: Configuration de l'indicateur ADX
        """
        self.config = config
        self.values: List[float] = []
        self.plus_di: List[float] = []
        self.minus_di: List[float] = []
        self.tr: List[float] = []
        self.dx: List[float] = []
        
    def calculate(self, high: List[float], low: List[float], close: List[float]) -> Dict[str, float]:
        """
        Calcule les valeurs de l'ADX.
        
        Args:
            high: Liste des prix hauts
            low: Liste des prix bas
            close: Liste des prix de clôture
            
        Returns:
            Dict[str, float]: Valeurs de l'ADX (adx, plus_di, minus_di)
        """
        if len(high) < self.config.period + 1:
            return {'adx': 50.0, 'plus_di': 50.0, 'minus_di': 50.0}
        
        # Calcul du True Range (TR)
        tr = []
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr.append(max(hl, hc, lc))
        
        # Calcul des mouvements directionnels
        plus_dm = []
        minus_dm = []
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
        
        # Calcul des moyennes
        tr_period = sum(tr[-self.config.period:]) / self.config.period
        plus_dm_period = sum(plus_dm[-self.config.period:]) / self.config.period
        minus_dm_period = sum(minus_dm[-self.config.period:]) / self.config.period
        
        # Calcul des indicateurs directionnels
        plus_di = 100 * plus_dm_period / tr_period if tr_period > 0 else 0
        minus_di = 100 * minus_dm_period / tr_period if tr_period > 0 else 0
        
        # Calcul du DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        # Calcul de l'ADX
        if len(self.values) == 0:
            adx = dx
        else:
            adx = (self.values[-1] * (self.config.period - 1) + dx) / self.config.period
        
        self.values.append(adx)
        self.plus_di.append(plus_di)
        self.minus_di.append(minus_di)
        self.tr.append(tr_period)
        self.dx.append(dx)
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def is_strong_trend(self) -> bool:
        """Vérifie si la tendance est forte."""
        return len(self.values) > 0 and self.values[-1] >= self.config.threshold
    
    def is_bullish_trend(self) -> bool:
        """Vérifie si la tendance est haussière."""
        if len(self.plus_di) == 0 or len(self.minus_di) == 0:
            return False
        return self.plus_di[-1] > self.minus_di[-1]
    
    def is_bearish_trend(self) -> bool:
        """Vérifie si la tendance est baissière."""
        if len(self.plus_di) == 0 or len(self.minus_di) == 0:
            return False
        return self.minus_di[-1] > self.plus_di[-1] 