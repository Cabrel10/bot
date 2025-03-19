from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import asyncio
import yaml
from dataclasses import dataclass

from ..data.data_types import PositionData, OrderData, TradeData
from ..utils.logger import TradingLogger
from .risk_manager import RiskManager
from .order_manager import OrderManager, OrderType

@dataclass
class PositionInfo:
    """Information sur une position."""
    symbol: str
    side: str  # 'long' ou 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    timestamp: datetime = datetime.now()

class PositionManager:
    """Gestionnaire des positions de trading."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialise le gestionnaire de positions.
        
        Args:
            config_path: Chemin vers la configuration
        """
        self.logger = TradingLogger()
        self.config = self._load_config(config_path)
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager()
        
        # État des positions
        self.positions: Dict[str, PositionInfo] = {}
        self.position_history: List[PositionData] = []
        
        # Suivi des performances
        self.initial_equity = self.config['account']['initial_equity']
        self.current_equity = self.initial_equity

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Charge la configuration."""
        if not config_path:
            config_path = Path(__file__).parent / 'config' / 'position_config.yaml'
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def open_position(self,
                          symbol: str,
                          side: str,
                          size: float,
                          price: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Optional[PositionInfo]:
        """Ouvre une nouvelle position.
        
        Args:
            symbol: Symbole à trader
            side: 'long' ou 'short'
            size: Taille de la position
            price: Prix d'entrée
            stop_loss: Prix du stop loss
            take_profit: Prix du take profit
            
        Returns:
            Information sur la position si créée
        """
        try:
            # Vérification des règles de risque
            allowed, reason = await self.risk_manager.check_order_risk(
                symbol=symbol,
                side=side,
                amount=size,
                price=price
            )
            
            if not allowed:
                self.logger.log_warning(f"Position rejetée: {reason}")
                return None
            
            # Création de l'ordre d'entrée
            order = await self.order_manager.create_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=side,
                amount=size,
                price=price
            )
            
            if order.status != "FILLED":
                return None
            
            # Création de la position
            position = PositionInfo(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Ajout aux positions actives
            self.positions[symbol] = position
            
            # Création des ordres de stop loss et take profit
            await self._create_exit_orders(position)
            
            return position

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'open_position',
                'symbol': symbol
            })
            return None

    async def close_position(self,
                           symbol: str,
                           reason: str = "manual") -> Optional[float]:
        """Ferme une position existante.
        
        Args:
            symbol: Symbole de la position
            reason: Raison de la fermeture
            
        Returns:
            PnL réalisé si la position est fermée
        """
        try:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            
            # Création de l'ordre de fermeture
            close_side = "sell" if position.side == "long" else "buy"
            order = await self.order_manager.create_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=close_side,
                amount=position.size
            )
            
            if order.status != "FILLED":
                return None
            
            # Calcul du PnL
            exit_price = order.price
            pnl = self._calculate_pnl(position, exit_price)
            
            # Mise à jour de l'equity
            self.current_equity += pnl
            
            # Archivage de la position
            position_data = PositionData(
                symbol=symbol,
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                side=position.side,
                entry_time=position.timestamp,
                exit_time=datetime.now(),
                pnl=pnl,
                reason=reason
            )
            self.position_history.append(position_data)
            
            # Suppression de la position active
            del self.positions[symbol]
            
            return pnl

        except Exception as e:
            self.logger.log_error(e, {
                'action': 'close_position',
                'symbol': symbol
            })
            return None

    async def update_positions(self, market_data: Dict[str, float]) -> None:
        """Met à jour les positions avec les derniers prix.
        
        Args:
            market_data: Prix actuels par symbole
        """
        try:
            for symbol, position in list(self.positions.items()):
                if symbol not in market_data:
                    continue
                
                current_price = market_data[symbol]
                position.current_price = current_price
                
                # Mise à jour du PnL non réalisé
                position.unrealized_pnl = self._calculate_pnl(position, current_price)
                
                # Vérification des conditions de sortie
                if self._check_exit_conditions(position):
                    await self.close_position(symbol, "auto_exit")
                
                # Mise à jour des stops trailing
                await self._update_trailing_stops(position)

        except Exception as e:
            self.logger.log_error(e, {'action': 'update_positions'})

    def get_position_summary(self) -> Dict[str, Any]:
        """Génère un résumé des positions actuelles."""
        return {
            "total_positions": len(self.positions),
            "total_exposure": self._calculate_total_exposure(),
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
            "realized_pnl": sum(p.realized_pnl for p in self.positions.values()),
            "current_equity": self.current_equity,
            "positions": [
                {
                    "symbol": symbol,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            ]
        }

    def _calculate_pnl(self, position: PositionInfo, current_price: float) -> float:
        """Calcule le PnL d'une position."""
        price_diff = current_price - position.entry_price
        if position.side == "short":
            price_diff = -price_diff
        return price_diff * position.size * position.leverage

    def _calculate_total_exposure(self) -> float:
        """Calcule l'exposition totale."""
        return sum(
            abs(pos.size * pos.current_price * pos.leverage)
            for pos in self.positions.values()
        )

    async def _create_exit_orders(self, position: PositionInfo) -> None:
        """Crée les ordres de sortie pour une position."""
        try:
            if position.stop_loss:
                await self.order_manager.create_order(
                    symbol=position.symbol,
                    order_type=OrderType.STOP_LOSS,
                    side="sell" if position.side == "long" else "buy",
                    amount=position.size,
                    price=position.stop_loss
                )
            
            if position.take_profit:
                await self.order_manager.create_order(
                    symbol=position.symbol,
                    order_type=OrderType.TAKE_PROFIT,
                    side="sell" if position.side == "long" else "buy",
                    amount=position.size,
                    price=position.take_profit
                )

        except Exception as e:
            self.logger.log_error(e, {'action': 'create_exit_orders'})

    def _check_exit_conditions(self, position: PositionInfo) -> bool:
        """Vérifie les conditions de sortie d'une position."""
        # Stop loss
        if position.stop_loss:
            if position.side == "long" and position.current_price <= position.stop_loss:
                return True
            if position.side == "short" and position.current_price >= position.stop_loss:
                return True
        
        # Take profit
        if position.take_profit:
            if position.side == "long" and position.current_price >= position.take_profit:
                return True
            if position.side == "short" and position.current_price <= position.take_profit:
                return True
        
        return False

    async def _update_trailing_stops(self, position: PositionInfo) -> None:
        """Met à jour les stops trailing."""
        if not self.config['stops']['enable_trailing']:
            return
        
        try:
            trailing_distance = self.config['stops']['trailing_distance']
            
            if position.side == "long":
                new_stop = position.current_price * (1 - trailing_distance)
                if not position.stop_loss or new_stop > position.stop_loss:
                    position.stop_loss = new_stop
            else:
                new_stop = position.current_price * (1 + trailing_distance)
                if not position.stop_loss or new_stop < position.stop_loss:
                    position.stop_loss = new_stop

        except Exception as e:
            self.logger.log_error(e, {'action': 'update_trailing_stops'})

# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Création du gestionnaire
        position_manager = PositionManager()
        
        try:
            # Ouverture d'une position
            position = await position_manager.open_position(
                symbol="BTC/USDT",
                side="long",
                size=0.1,
                price=50000,
                stop_loss=49000,
                take_profit=52000
            )
            
            if position:
                print(f"Position ouverte: {position}")
                
                # Simulation de mise à jour des prix
                await position_manager.update_positions({
                    "BTC/USDT": 51000
                })
                
                # Affichage du résumé
                print("\nRésumé des positions:")
                print(position_manager.get_position_summary())
                
                # Fermeture de la position
                pnl = await position_manager.close_position("BTC/USDT")
                print(f"\nPnL réalisé: {pnl}")
            
        except Exception as e:
            print(f"Erreur: {e}")

    # Exécution
    asyncio.run(main())