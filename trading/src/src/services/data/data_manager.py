"""Service de gestion des données de trading."""
import asyncio
import redis
from aiohttp import ClientSession
from trading.utils.logging import TradingLogger
from trading.core.data import MarketDataClient
from typing import Optional, Dict

class DataManager:
    """Gestionnaire des données de trading."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = TradingLogger()
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
        
        # Passer la configuration au MarketDataClient
        self.market_client = MarketDataClient(config)  # Ajoutez config ici
        
        self._running = False
        self._task = None
    
    def start(self):
        """Démarre le service de données."""
        if self._running:
            self.logger.warning("Le service est déjà en cours d'exécution")
            return
            
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self.fetch_market_data())
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Arrête le service de données."""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    
    async def fetch_market_data(self):
        """Récupère les données de marché."""
        async with ClientSession() as session:
            while self._running:
                try:
                    data = await self.market_client.fetch_data(session)
                    self.redis_client.set('market_data', data)
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données: {e}")
                    await asyncio.sleep(5)
                    
if __name__ == '__main__':
    data_manager = DataManager()
    try:
        data_manager.start()
    except KeyboardInterrupt:
        data_manager.stop()
