import pytest
from src.execution.order_manager import OrderManager, Order, OrderStatus
from src.security.api_security import APISecurityManager
from src.risk.risk_manager import RiskManager
from datetime import datetime

@pytest.fixture
def setup_execution_environment():
    """Prépare l'environnement de test."""
    api_security = APISecurityManager()
    risk_manager = RiskManager({
        'max_position_size': 1.0,
        'max_drawdown': 0.1,
        'exposure_limits': {'BTC': 1.0}
    })
    order_manager = OrderManager(risk_manager, api_security)
    return {
        'order_manager': order_manager,
        'risk_manager': risk_manager,
        'api_security': api_security
    }

class TestOrderExecution:
    def test_order_creation(self, setup_execution_environment):
        """Teste la création d'ordre."""
        order_manager = setup_execution_environment['order_manager']
        
        order = order_manager.create_order(
            symbol="BTC/USD",
            side="BUY",
            quantity=0.1,
            order_type="MARKET"
        )
        
        assert order.symbol == "BTC/USD"
        assert order.status == OrderStatus.PENDING
        
    def test_order_validation(self, setup_execution_environment):
        """Teste la validation des ordres."""
        order_manager = setup_execution_environment['order_manager']
        
        # Test ordre valide
        valid_order = Order(
            symbol="BTC/USD",
            side="BUY",
            quantity=0.1,
            order_type="MARKET"
        )
        assert order_manager.validate_order(valid_order)
        
        # Test ordre invalide
        with pytest.raises(ValueError):
            invalid_order = Order(
                symbol="BTC/USD",
                side="INVALID",
                quantity=0.1,
                order_type="MARKET"
            )
            order_manager.validate_order(invalid_order)
            
    def test_order_submission(self, setup_execution_environment):
        """Teste la soumission d'ordre."""
        order_manager = setup_execution_environment['order_manager']
        
        order = order_manager.create_order(
            symbol="BTC/USD",
            side="BUY",
            quantity=0.1,
            order_type="MARKET"
        )
        
        submitted_order = order_manager.submit_order(order)
        assert submitted_order.status == OrderStatus.FILLED
        
    def test_order_cancellation(self, setup_execution_environment):
        """Teste l'annulation d'ordre."""
        order_manager = setup_execution_environment['order_manager']
        
        order = order_manager.create_order(
            symbol="BTC/USD",
            side="BUY",
            quantity=0.1,
            order_type="LIMIT",
            price=50000
        )
        
        submitted_order = order_manager.submit_order(order)
        assert order_manager.cancel_order(submitted_order.order_id)
        
    def test_api_security(self, setup_execution_environment):
        """Teste la sécurité des API."""
        api_security = setup_execution_environment['api_security']
        
        # Test ajout de clés
        api_security.add_api_key(
            "binance",
            "test_api_key",
            "test_api_secret"
        )
        
        # Test récupération de clés
        creds = api_security.get_api_credentials("binance")
        assert creds['api_key'] == "test_api_key"
        assert creds['api_secret'] == "test_api_secret"
        
        # Test suppression de clés
        assert api_security.remove_api_key("binance")

