from typing import Iterator, Optional, Dict
import pandas as pd
from ..core.data_types import MarketData
from ..utils.memory_manager import MemoryManager
from .backtester import Backtester

class StreamingBacktester(Backtester):
    """Backtester avec streaming de données."""
    
    def __init__(self, chunk_size: int = 1000,
                 max_memory_usage: float = 0.8):
        """
        Args:
            chunk_size: Taille des chunks de données
            max_memory_usage: Utilisation mémoire maximale
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager(max_memory_usage)
        
    def _data_generator(self, data: MarketData) -> Iterator[MarketData]:
        """Génère des chunks de données.
        
        Args:
            data: Données complètes
            
        Yields:
            MarketData: Chunk de données
        """
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i+self.chunk_size]
            yield chunk
            
    def run_backtest(self, model, data: MarketData,
                    **kwargs) -> Dict:
        """Execute le backtest en streaming.
        
        Args:
            model: Modèle à tester
            data: Données de test
            
        Returns:
            Dict: Résultats du backtest
        """
        results = {
            'trades': [],
            'metrics': {},
            'equity_curve': []
        }
        
        for chunk in self._data_generator(data):
            # Vérifie la mémoire
            self.memory_manager.optimize_memory()
            
            # Traite le chunk
            chunk_results = super().run_backtest(model, chunk, **kwargs)
            
            # Agrège les résultats
            results['trades'].extend(chunk_results['trades'])
            results['equity_curve'].extend(chunk_results['equity_curve'])
            
        # Calcule les métriques finales
        results['metrics'] = self._calculate_final_metrics(results)
        return results 