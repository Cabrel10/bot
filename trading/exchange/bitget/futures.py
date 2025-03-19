"""
Module client Bitget Futures.

Ce module implémente le client Bitget pour le trading de contrats à terme,
fournissant toutes les fonctionnalités nécessaires pour interagir
avec l'API futures de Bitget.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from .client import BitgetClientBase
from .constants import (
    BITGET_FUTURES_MAINNET,
    BITGET_FUTURES_TESTNET,
    BITGET_WS_FUTURES_MAINNET,
    BITGET_WS_FUTURES_TESTNET,
    ENDPOINTS
)
from .utils import (
    validate_symbol,
    validate_interval,
    validate_order_type,
    validate_order_side,
    validate_position_side,
    validate_margin_type,
    parse_timestamp,
    format_number,
    parse_order_status
)
from ..base import (
    OrderRequest,
    OrderResponse,
    AccountBalance,
    MarketInfo,
    Position
)
from .errors import BitgetAPIError

class BitgetFuturesClient(BitgetClientBase):
    """Client pour le trading de contrats à terme sur Bitget."""
    
    @property
    def base_url(self) -> str:
        """URL de base de l'API futures."""
        return BITGET_FUTURES_TESTNET if self.testnet else BITGET_FUTURES_MAINNET
        
    @property
    def ws_url(self) -> str:
        """URL de base du WebSocket futures."""
        return BITGET_WS_FUTURES_TESTNET if self.testnet else BITGET_WS_FUTURES_MAINNET
        
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Récupère les informations de l'exchange.
        
        Returns:
            Informations de l'exchange
        """
        return await self._request('GET', ENDPOINTS['futures_exchange_info'])
        
    async def get_market_info(self, symbol: str) -> MarketInfo:
        """
        Récupère les informations d'un marché.
        
        Args:
            symbol: Symbole du marché
            
        Returns:
            Informations du marché
        """
        symbol = validate_symbol(symbol)
        info = await self.get_exchange_info()
        
        for market in info['data']:
            if market['symbol'] == symbol:
                return MarketInfo(
                    symbol=symbol,
                    base_asset=market['baseCoin'],
                    quote_asset=market['quoteCoin'],
                    price_precision=market['pricePrecision'],
                    quantity_precision=market['quantityPrecision'],
                    min_quantity=float(market['minTradeAmount']),
                    max_quantity=float(market.get('maxTradeAmount', 0)),
                    min_notional=float(market['minTradeUSDT']),
                    raw_info=market
                )
                
        raise ValueError(f"Symbole non trouvé: {symbol}")
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le ticker d'un symbole.
        
        Args:
            symbol: Symbole du marché
            
        Returns:
            Ticker du symbole
        """
        symbol = validate_symbol(symbol)
        return await self._request(
            'GET',
            ENDPOINTS['futures_ticker'],
            {'symbol': symbol}
        )
        
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres d'un symbole.
        
        Args:
            symbol: Symbole du marché
            limit: Nombre de niveaux à récupérer
            
        Returns:
            Carnet d'ordres
        """
        symbol = validate_symbol(symbol)
        return await self._request(
            'GET',
            ENDPOINTS['futures_depth'],
            {
                'symbol': symbol,
                'limit': limit
            }
        )
        
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Récupère les trades récents d'un symbole.
        
        Args:
            symbol: Symbole du marché
            limit: Nombre de trades à récupérer
            
        Returns:
            Liste des trades
        """
        symbol = validate_symbol(symbol)
        return await self._request(
            'GET',
            ENDPOINTS['futures_trades'],
            {
                'symbol': symbol,
                'limit': limit
            }
        )
        
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère les chandeliers d'un symbole.
        
        Args:
            symbol: Symbole du marché
            interval: Intervalle des chandeliers
            start_time: Début de la période
            end_time: Fin de la période
            limit: Nombre de chandeliers à récupérer
            
        Returns:
            Liste des chandeliers
        """
        symbol = validate_symbol(symbol)
        interval = validate_interval(interval)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = parse_timestamp(start_time)
        if end_time:
            params['endTime'] = parse_timestamp(end_time)
            
        return await self._request('GET', ENDPOINTS['futures_klines'], params)
        
    async def create_order(self, order: OrderRequest) -> OrderResponse:
        """
        Crée un ordre.
        
        Args:
            order: Requête de création d'ordre
            
        Returns:
            Réponse de création d'ordre
        """
        symbol = validate_symbol(order.symbol)
        side = validate_order_side(order.side)
        type_ = validate_order_type(order.type)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': type_,
            'quantity': format_number(order.quantity)
        }
        
        if order.price:
            params['price'] = format_number(order.price)
            
        if order.client_order_id:
            params['clientOrderId'] = order.client_order_id
            
        if order.reduce_only is not None:
            params['reduceOnly'] = order.reduce_only
            
        if order.position_side:
            params['positionSide'] = validate_position_side(order.position_side)
            
        response = await self._request(
            'POST',
            ENDPOINTS['futures_order'],
            data=params,
            signed=True
        )
        
        return OrderResponse(
            exchange_order_id=str(response['orderId']),
            client_order_id=response.get('clientOrderId'),
            symbol=response['symbol'],
            status=parse_order_status(response['status']),
            side=response['side'],
            type=response['type'],
            quantity=float(response['quantity']),
            price=float(response['price']),
            filled_quantity=float(response['filledQuantity']),
            remaining_quantity=float(response['quantity']) - float(response['filledQuantity']),
            average_price=float(response.get('avgPrice', 0)),
            commission=float(response.get('fee', 0)),
            commission_asset=response.get('feeAsset'),
            created_at=datetime.fromtimestamp(response['createTime'] / 1000),
            updated_at=datetime.fromtimestamp(response['updateTime'] / 1000),
            raw_response=response
        )
        
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Annule un ordre.
        
        Args:
            symbol: Symbole du marché
            order_id: ID de l'ordre
            client_order_id: ID client de l'ordre
            
        Returns:
            Réponse d'annulation
        """
        symbol = validate_symbol(symbol)
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['clientOrderId'] = client_order_id
        else:
            raise ValueError("order_id ou client_order_id requis")
            
        return await self._request(
            'POST',
            ENDPOINTS['futures_cancel_order'],
            data=params,
            signed=True
        )
        
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> OrderResponse:
        """
        Récupère un ordre.
        
        Args:
            symbol: Symbole du marché
            order_id: ID de l'ordre
            client_order_id: ID client de l'ordre
            
        Returns:
            Informations de l'ordre
        """
        symbol = validate_symbol(symbol)
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['clientOrderId'] = client_order_id
        else:
            raise ValueError("order_id ou client_order_id requis")
            
        response = await self._request(
            'GET',
            ENDPOINTS['futures_order'],
            params=params,
            signed=True
        )
        
        return OrderResponse(
            exchange_order_id=str(response['orderId']),
            client_order_id=response.get('clientOrderId'),
            symbol=response['symbol'],
            status=parse_order_status(response['status']),
            side=response['side'],
            type=response['type'],
            quantity=float(response['quantity']),
            price=float(response['price']),
            filled_quantity=float(response['filledQuantity']),
            remaining_quantity=float(response['quantity']) - float(response['filledQuantity']),
            average_price=float(response.get('avgPrice', 0)),
            commission=float(response.get('fee', 0)),
            commission_asset=response.get('feeAsset'),
            created_at=datetime.fromtimestamp(response['createTime'] / 1000),
            updated_at=datetime.fromtimestamp(response['updateTime'] / 1000),
            raw_response=response
        )
        
    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[OrderResponse]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole du marché (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        params = {}
        if symbol:
            params['symbol'] = validate_symbol(symbol)
            
        response = await self._request(
            'GET',
            ENDPOINTS['futures_open_orders'],
            params=params,
            signed=True
        )
        
        return [
            OrderResponse(
                exchange_order_id=str(order['orderId']),
                client_order_id=order.get('clientOrderId'),
                symbol=order['symbol'],
                status=parse_order_status(order['status']),
                side=order['side'],
                type=order['type'],
                quantity=float(order['quantity']),
                price=float(order['price']),
                filled_quantity=float(order['filledQuantity']),
                remaining_quantity=float(order['quantity']) - float(order['filledQuantity']),
                average_price=float(order.get('avgPrice', 0)),
                commission=float(order.get('fee', 0)),
                commission_asset=order.get('feeAsset'),
                created_at=datetime.fromtimestamp(order['createTime'] / 1000),
                updated_at=datetime.fromtimestamp(order['updateTime'] / 1000),
                raw_response=order
            )
            for order in response
        ]
        
    async def get_account_balance(self) -> List[AccountBalance]:
        """
        Récupère le solde du compte.
        
        Returns:
            Liste des soldes par actif
        """
        response = await self._request(
            'GET',
            ENDPOINTS['futures_account'],
            signed=True
        )
        
        return [
            AccountBalance(
                asset=balance['coin'],
                free=float(balance['available']),
                locked=float(balance['frozen']),
                total=float(balance['total']),
                raw_balance=balance
            )
            for balance in response
        ]
        
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[Position]:
        """
        Récupère les positions ouvertes.
        
        Args:
            symbol: Symbole du marché (optionnel)
            
        Returns:
            Liste des positions
        """
        params = {}
        if symbol:
            params['symbol'] = validate_symbol(symbol)
            
        response = await self._request(
            'GET',
            ENDPOINTS['futures_positions'],
            params=params,
            signed=True
        )
        
        return [
            Position(
                symbol=position['symbol'],
                side=position['side'],
                quantity=float(position['quantity']),
                entry_price=float(position['entryPrice']),
                unrealized_pnl=float(position['unrealizedPnl']),
                leverage=int(position['leverage']),
                margin_type=position['marginType'],
                raw_position=position
            )
            for position in response
        ]
        
    async def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> Dict[str, Any]:
        """
        Définit l'effet de levier pour un symbole.
        
        Args:
            symbol: Symbole du marché
            leverage: Effet de levier (1-125)
            
        Returns:
            Réponse de l'API
        """
        symbol = validate_symbol(symbol)
        
        if not isinstance(leverage, int) or leverage < 1 or leverage > 125:
            raise ValueError("L'effet de levier doit être un entier entre 1 et 125")
            
        return await self._request(
            'POST',
            ENDPOINTS['futures_leverage'],
            data={
                'symbol': symbol,
                'leverage': leverage
            },
            signed=True
        )
        
    async def set_margin_type(
        self,
        symbol: str,
        margin_type: str
    ) -> Dict[str, Any]:
        """
        Définit le type de marge pour un symbole.
        
        Args:
            symbol: Symbole du marché
            margin_type: Type de marge (ISOLATED ou CROSSED)
            
        Returns:
            Réponse de l'API
        """
        symbol = validate_symbol(symbol)
        margin_type = validate_margin_type(margin_type)
        
        return await self._request(
            'POST',
            ENDPOINTS['futures_margin_type'],
            data={
                'symbol': symbol,
                'marginType': margin_type
            },
            signed=True
        )
        
    async def subscribe_ticker(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour du ticker.
        
        Args:
            symbol: Symbole du marché
        """
        symbol = validate_symbol(symbol).lower()
        stream = f"futures/ticker:{symbol}"
        self.ws_manager = await self._create_ws_connection(
            stream,
            self._handle_market_data_callback
        )
        
    async def subscribe_orderbook(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour du carnet d'ordres.
        
        Args:
            symbol: Symbole du marché
        """
        symbol = validate_symbol(symbol).lower()
        stream = f"futures/depth:{symbol}"
        self.ws_manager = await self._create_ws_connection(
            stream,
            self._handle_market_data_callback
        )
        
    async def subscribe_trades(self, symbol: str) -> None:
        """
        Souscrit aux mises à jour des trades.
        
        Args:
            symbol: Symbole du marché
        """
        symbol = validate_symbol(symbol).lower()
        stream = f"futures/trade:{symbol}"
        self.ws_manager = await self._create_ws_connection(
            stream,
            self._handle_market_data_callback
        )
        
    async def subscribe_klines(self, symbol: str, interval: str) -> None:
        """
        Souscrit aux mises à jour des chandeliers.
        
        Args:
            symbol: Symbole du marché
            interval: Intervalle des chandeliers
        """
        symbol = validate_symbol(symbol).lower()
        interval = validate_interval(interval)
        stream = f"futures/candle{interval}:{symbol}"
        self.ws_manager = await self._create_ws_connection(
            stream,
            self._handle_market_data_callback
        )
        
    async def subscribe_user_data(self) -> None:
        """
        Souscrit aux mises à jour des données utilisateur.
        """
        streams = [
            "futures/account",
            "futures/positions",
            "futures/orders"
        ]
        
        for stream in streams:
            self.ws_manager = await self._create_ws_connection(
                stream,
                self._handle_user_data_callback
            )
            
    def _handle_user_data_callback(self, msg: Dict[str, Any]) -> None:
        """
        Gère les messages de données utilisateur.
        
        Args:
            msg: Message WebSocket
        """
        channel = msg.get('arg', {}).get('channel', '')
        
        if channel == 'futures/account':
            # Mise à jour des soldes
            pass
        elif channel == 'futures/positions':
            # Mise à jour des positions
            pass
        elif channel == 'futures/orders':
            # Mise à jour des ordres
            pass
            
    def _handle_market_data_callback(self, msg: Dict[str, Any]) -> None:
        """
        Gère les messages de données de marché.
        
        Args:
            msg: Message WebSocket
        """
        channel = msg.get('arg', {}).get('channel', '')
        
        if 'ticker' in channel:
            # Mise à jour du ticker
            pass
        elif 'depth' in channel:
            # Mise à jour du carnet d'ordres
            pass
        elif 'trade' in channel:
            # Mise à jour des trades
            pass
        elif 'candle' in channel:
            # Mise à jour des chandeliers
            pass 