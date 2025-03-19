from historical_data import HistoricalDataManager
from dataset_manager import DatasetManager
from datetime import datetime
import asyncio

async def process_trading_data():
    # Initialisation
    hist_manager = HistoricalDataManager()
    dataset_manager = DatasetManager()
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    start_date = datetime(2024, 1, 1)
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                # Récupération des données
                df = await hist_manager.update_historical_data(
                    symbol, timeframe, start_date
                )
                
                # Traitement et validation des données
                processed_df = dataset_manager.process_data(df, symbol, timeframe)
                
                print(f"Données traitées avec succès pour {symbol} - {timeframe}")
                
            except Exception as e:
                print(f"Erreur pour {symbol} - {timeframe}: {str(e)}")
                continue

if __name__ == "__main__":
    asyncio.run(process_trading_data()) 