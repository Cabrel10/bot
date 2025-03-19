"""
Module client Binance Futures.

Ce module implémente le client Binance pour le trading de contrats à terme,
fournissant toutes les fonctionnalités nécessaires pour interagir avec
l'API futures de Binance.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from .client import BinanceClientBase
from .constants import (
    BINANCE_FUTURES_MAINNET,
    BINANCE_FUTURES_TESTNET,
    BINANCE_WS_FUTURES_MAINNET,
    BINANCE_WS_FUTURES_TESTNET,
    ENDPOINTS,
    MARGIN_TYPE,
    POSITION_SIDE
)
from .utils import (
    validate_symbol,
    validate_interval,
    validate_order_type,
    validate_order_side,
    validate_time_in_force,
    parse_timestamp,
    format_number,
    parse_order_status
)
from ..base import (
    OrderRequest,
    OrderResponse,
    AccountBalance,
    MarketInfo,
    PositionInfo
)
from ..errors import InvalidOrderError

class BinanceFuturesClient(BinanceClientBase):
    """Client pour le trading futures sur Binance."""
    
    @property
    def base_url(self) -> str:
        """URL de base de l'API futures."""
        return BINANCE_FUTURES_TESTNET if self.testnet else BINANCE_FUTURES_MAINNET
        
    @property
    def ws_url(self) -> str:
        """URL de base du WebSocket futures."""
        return BINANCE_WS_FUTURES_TESTNET if self.testnet else BINANCE_WS_FUTURES_MAINNET
        
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
        
        for market in info['symbols']:
            if market['symbol'] == symbol:
                filters = {f['filterType']: f for f in market['filters']}
                
                return MarketInfo(
                    symbol=symbol,
                    base_asset=market['baseAsset'],
                    quote_asset=market['quoteAsset'],
                    price_precision=market['pricePrecision'],
                    quantity_precision=market['quantityPrecision'],
                    min_quantity=float(filters['LOT_SIZE']['minQty']),
                    max_quantity=float(filters['LOT_SIZE']['maxQty']),
                    min_notional=float(filters['MIN_NOTIONAL']['notional']),
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
        
    async def get_historical_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades d'un symbole.
        
        Args:
            symbol: Symbole du marché
            start_time: Début de la période
            end_time: Fin de la période
            limit: Nombre de trades à récupérer
            
        Returns:
            Liste des trades
        """
        symbol = validate_symbol(symbol)
        params = {'symbol': symbol, 'limit': limit}
        
        if start_time:
            params['startTime'] = parse_timestamp(start_time)
        if end_time:
            params['endTime'] = parse_timestamp(end_time)
            
        return await self._request(
            'GET',
            ENDPOINTS['futures_historical_trades'],
            params,
            signed=True
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
            
        Raises:
            InvalidOrderError: Si les paramètres de l'ordre sont invalides
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
            
        if order.time_in_force:
            params['timeInForce'] = validate_time_in_force(order.time_in_force)
            
        if order.stop_price:
            params['stopPrice'] = format_number(order.stop_price)
            
        if order.client_order_id:
            params['newClientOrderId'] = order.client_order_id
            
        if order.reduce_only:
            params['reduceOnly'] = 'true'
            
        if order.leverage:
            await self.set_leverage(symbol, order.leverage)
            
        response = await self._request(
            'POST',
            ENDPOINTS['futures_order'],
            params,
            signed=True
        )
        
        return OrderResponse(
            exchange_order_id=str(response['orderId']),
            client_order_id=response.get('clientOrderId'),
            symbol=response['symbol'],
            status=parse_order_status(response['status']),
            side=response['side'],
            type=response['type'],
            quantity=float(response['origQty']),
            price=float(response['price']),
            filled_quantity=float(response['executedQty']),
            remaining_quantity=float(response['origQty']) - float(response['executedQty']),
            average_price=float(response.get('avgPrice', 0)),
            commission=float(response.get('commission', 0)),
            commission_asset=response.get('commissionAsset'),
            created_at=datetime.fromtimestamp(response['time'] / 1000),
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
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("order_id ou client_order_id requis")
            
        return await self._request(
            'DELETE',
            ENDPOINTS['futures_order'],
            params,
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
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("order_id ou client_order_id requis")
            
        response = await self._request(
            'GET',
            ENDPOINTS['futures_order'],
            params,
            signed=True
        )
        
        return OrderResponse(
            exchange_order_id=str(response['orderId']),
            client_order_id=response.get('clientOrderId'),
            symbol=response['symbol'],
            status=parse_order_status(response['status']),
            side=response['side'],
            type=response['type'],
            quantity=float(response['origQty']),
            price=float(response['price']),
            filled_quantity=float(response['executedQty']),
            remaining_quantity=float(response['origQty']) - float(response['executedQty']),
            average_price=float(response.get('avgPrice', 0)),
            commission=float(response.get('commission', 0)),
            commission_asset=response.get('commissionAsset'),
            created_at=datetime.fromtimestamp(response['time'] / 1000),
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
            params,
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
                quantity=float(order['origQty']),
                price=float(order['price']),
                filled_quantity=float(order['executedQty']),
                remaining_quantity=float(order['origQty']) - float(order['executedQty']),
                average_price=float(order.get('avgPrice', 0)),
                commission=float(order.get('commission', 0)),
                commission_asset=order.get('commissionAsset'),
                created_at=datetime.fromtimestamp(order['time'] / 1000),
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
            ENDPOINTS['futures_balance'],
            signed=True
        )
        
        return [
            AccountBalance(
                asset=balance['asset'],
                free=float(balance['availableBalance']),
                locked=float(balance['balance']) - float(balance['availableBalance']),
                total=float(balance['balance']),
                raw_balance=balance
            )
            for balance in response
        ]
        
    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[PositionInfo]:
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
            ENDPOINTS['futures_position_risk'],
            params,
            signed=True
        )
        
        return [
            PositionInfo(
                symbol=position['symbol'],
                side='LONG' if float(position['positionAmt']) > 0 else 'SHORT',
                quantity=abs(float(position['positionAmt'])),
                entry_price=float(position['entryPrice']),
                leverage=int(position['leverage']),
                unrealized_pnl=float(position['unRealizedProfit']),
                margin_type=position['marginType'],
                liquidation_price=float(position.get('liquidationPrice', 0)),
                raw_info=position
            )
            for position in response
            if float(position['positionAmt']) != 0
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
            leverage: Effet de levier
            
        Returns:
            Réponse de l'exchange
        """
        symbol = validate_symbol(symbol)
        return await self._request(
            'POST',
            ENDPOINTS['futures_leverage'],
            {
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
            margin_type: Type de marge ('ISOLATED' ou 'CROSSED')
            
        Returns:
            Réponse de l'exchange
        """
        symbol = validate_symbol(symbol)
        margin_type = margin_type.upper()
        
        if margin_type not in MARGIN_TYPE:
            raise ValueError(
                f"Type de marge invalide. Valeurs possibles: {', '.join(MARGIN_TYPE)}"
            )
            
        return await self._request(
            'POST',
            ENDPOINTS['futures_margin_type'],
            {
                'symbol': symbol,
                'marginType': MARGIN_TYPE[margin_type]
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
        stream = f"{symbol}@ticker"
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
        stream = f"{symbol}@depth"
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
        stream = f"{symbol}@trade"
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
        stream = f"{symbol}@kline_{interval}"
        self.ws_manager = await self._create_ws_connection(
            stream,
            self._handle_market_data_callback
        )
        
    async def subscribe_user_data(self) -> None:
        """
        Souscrit aux mises à jour des données utilisateur.
        
        Raises:
            AuthenticationError: Si les clés API ne sont pas configurées
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required")
            
        # Obtention de la clé d'écoute
        response = await self._request(
            'POST',
            '/fapi/v1/listenKey',
            signed=False
        )
        
        listen_key = response['listenKey']
        stream = f"{listen_key}"
        
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
        event_type = msg.get('e')
        if event_type == 'ACCOUNT_UPDATE':
            # Mise à jour des soldes et positions
            pass
        elif event_type == 'ORDER_TRADE_UPDATE':
            # Mise à jour des ordres
            pass
        elif event_type == 'MARGIN_CALL':
            # Appel de marge
            pass
            
    def _handle_market_data_callback(self, msg: Dict[str, Any]) -> None:
        """
        Gère les messages de données de marché.
        
        Args:
            msg: Message WebSocket
        """
        stream = msg.get('stream', '')
        
        if '@ticker' in stream:
            # Mise à jour du ticker
            pass
        elif '@depth' in stream:
            # Mise à jour du carnet d'ordres
            pass
        elif '@trade' in stream:
            # Mise à jour des trades
            pass
        elif '@kline' in stream:
            # Mise à jour des chandeliers
            pass 