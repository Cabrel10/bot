from typing import Dict, List, Optional, Union, Any
from enum import Enum
import asyncio
from datetime import datetime
import uuid
from pathlib import Path
import yaml
import json
from collections import deque

from ..data.exchange_api import ExchangeAPI
from ..data.data_types import OrderData, TradeData, ProcessedData
from ..utils.logger import TradingLogger

class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Statuts possibles d'un ordre."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderManager:
    """Gestionnaire des ordres de trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise le gestionnaire d'ordres.
        
        Args:
            config_path: Chemin vers la configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.exchange = ExchangeAPI()
        
        # Files d'attente des ordres
        self.order_queue = deque()
        self.active_orders: Dict[str, OrderData] = {}
        self.completed_orders: List[OrderData] = []
        
        # Limites et contrôles
        self.rate_limits = self.config['rate_limits']
        self.last_order_time = datetime.now()

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Charge la configuration."""
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'order_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def create_order(self,
                          symbol: str,
                          order_type: OrderType,
                          side: str,
                          amount: float,
                          price: Optional[float] = None,
                          params: Optional[Dict[str, Any]] = None) -> OrderData:
        """Crée un nouvel ordre.
        
        Args:
            symbol: Symbole à trader
            order_type: Type d'ordre
            side: 'buy' ou 'sell'
            amount: Quantité à trader
            price: Prix pour les ordres limit
            params: Paramètres additionnels
            
        Returns:
            Données de l'ordre créé
        """
        try:
            # Validation des paramètres
            self._validate_order_params(symbol, order_type, side, amount, price)
            
            # Vérification des limites de taux
            await self._check_rate_limits()
            
            # Création de l'ordre
            order_id = str(uuid.uuid4())
            order = OrderData(
                id=order_id,
                symbol=symbol,
                type=order_type.value,
                side=side,
                amount=amount,
                price=price,
                status=OrderStatus.PENDING.value,
                timestamp=datetime.now(),
                params=params or {}
            )
            
            # Ajout à la file d'attente
            self.order_queue.append(order)
            
            # Traitement immédiat si possible
            if self._can_process_order():
                await self._process_order(order)
            
            return order

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'create_order',
                'symbol': symbol,
                'type': order_type.value
            })
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre actif.
        
        Args:
            order_id: ID de l'ordre à annuler
            
        Returns:
            True si l'annulation a réussi
        """
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Annulation sur l'exchange
                await self.exchange.cancel_order(order_id, order.symbol)
                
                # Mise à jour du statut
                order.status = OrderStatus.CANCELLED.value
                order.update_timestamp = datetime.now()
                
                # Déplacement vers les ordres complétés
                self.completed_orders.append(order)
                del self.active_orders[order_id]
                
                return True
            
            return False

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'cancel_order',
                'order_id': order_id
            })
            return False

    async def _process_order(self, order: OrderData) -> None:
        """Traite un ordre en attente."""
        try:
            # Envoi de l'ordre à l'exchange
            response = await self.exchange.create_order(
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                amount=order.amount,
                price=order.price,
                params=order.params
            )
            
            # Mise à jour avec la réponse de l'exchange
            order.exchange_id = response['id']
            order.status = OrderStatus.OPEN.value
            order.update_timestamp = datetime.now()
            
            # Ajout aux ordres actifs
            self.active_orders[order.id] = order
            
            # Mise à jour des limites de taux
            self.last_order_time = datetime.now()

        except Exception as e:
            order.status = OrderStatus.REJECTED.value
            order.error = str(e)
            self.completed_orders.append(order)
            self.logger.log_error(e, {
                'action': 'process_order',
                'order_id': order.id
            })

    async def update_orders(self) -> None:
        """Met à jour le statut des ordres actifs."""
        try:
            for order_id, order in list(self.active_orders.items()):
                # Récupération du statut depuis l'exchange
                status = await self.exchange.fetch_order_status(
                    order_id,
                    order.symbol
                )
                
                # Mise à jour du statut
                if status != order.status:
                    order.status = status
                    order.update_timestamp = datetime.now()
                    
                    # Déplacement vers les ordres complétés si terminé
                    if status in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
                        self.completed_orders.append(order)
                        del self.active_orders[order_id]

        except Exception as e:
            self.logger.log_error(e, {'action': 'update_orders'})

    def _validate_order_params(self,
                             symbol: str,
                             order_type: OrderType,
                             side: str,
                             amount: float,
                             price: Optional[float]) -> None:
        """Valide les paramètres d'un ordre."""
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
            
        if amount <= 0:
            raise ValueError(f"Invalid amount: {amount}")
            
        if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS] and not price:
            raise ValueError("Price required for limit/stop orders")

    async def _check_rate_limits(self) -> None:
        """Vérifie les limites de taux."""
        current_time = datetime.now()
        time_diff = (current_time - self.last_order_time).total_seconds()
        
        if time_diff < self.rate_limits['min_order_interval']:
            await asyncio.sleep(self.rate_limits['min_order_interval'] - time_diff)

    def _can_process_order(self) -> bool:
        """Vérifie si un nouvel ordre peut être traité."""
        return len(self.active_orders) < self.config['max_active_orders']

    def get_order_history(self, 
                         symbol: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[OrderData]:
        """Récupère l'historique des ordres avec filtres optionnels."""
        filtered_orders = self.completed_orders.copy()
        
        if symbol:
            filtered_orders = [o for o in filtered_orders if o.symbol == symbol]
        
        if start_time:
            filtered_orders = [o for o in filtered_orders if o.timestamp >= start_time]
        
        if end_time:
            filtered_orders = [o for o in filtered_orders if o.timestamp <= end_time]
        
        return filtered_orders

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création du gestionnaire
        manager = OrderManager()
        
        try:
            # Création d'un ordre
            order = await manager.create_order(
                symbol="BTC/USDT",
                order_type=OrderType.LIMIT,
                side="buy",
                amount=0.1,
                price=50000
            )
            
            print(f"Ordre créé: {order.id}")
            
            # Mise à jour des ordres
            await manager.update_orders()
            
            # Annulation si nécessaire
            if order.status == OrderStatus.OPEN.value:
                cancelled = await manager.cancel_order(order.id)
                print(f"Ordre annulé: {cancelled}")
            
        except Exception as e:
            print(f"Erreur: {e}")

    # Exécution
    asyncio.run(main()) 