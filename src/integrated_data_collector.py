#!/usr/bin/env python3
"""
Integrated Data Collector
Collects price data (10 years) + news sentiment and merges them
"""

import pandas as pd
import logging
from pathlib import Path
from ml_data_collector import MLDataCollector
from news_sentiment_collector import NewsSentimentCollector

logger = logging.getLogger(__name__)


class IntegratedDataCollector:
    """Collect and merge price data with sentiment data"""
    
    def __init__(self, newsapi_key, mt5_config=None):
        """
        Initialize integrated collector
        
        Args:
            newsapi_key: NewsAPI.org API key
            mt5_config: MT5 configuration (optional, uses default if None)
        """
        self.price_collector = MLDataCollector(mt5_config)
        self.sentiment_collector = NewsSentimentCollector(newsapi_key)
        self.output_dir = Path('ml_data_integrated')
        self.output_dir.mkdir(exist_ok=True)
        
    def merge_price_sentiment(self, price_df, sentiment_df, symbol):
        """
        Merge price data with sentiment data
        
        Args:
            price_df: Price data DataFrame (hourly)
            sentiment_df: Sentiment data DataFrame (daily)
            symbol: Trading symbol
            
        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging price and sentiment data for {symbol}...")
        
        # Convert price index to date (for merging)
        price_df = price_df.copy()
        price_df['date'] = price_df.index.date
        
        # Merge on date
        merged = price_df.merge(
            sentiment_df,
            on='date',
            how='left'
        )
        
        # Forward fill sentiment (use previous day's sentiment if no news)
        sentiment_cols = ['sentiment_score', 'news_count', 'sentiment_std',
                         'sentiment_ma_7', 'sentiment_ma_30', 'sentiment_change',
                         'news_volume_ma_7', 'major_news']
        
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(method='ffill').fillna(0)
        
        # Drop the temporary date column
        merged = merged.drop('date', axis=1)
        
        # Set index back to datetime
        merged.index = price_df.index
        
        logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
        logger.info(f"Sentiment features added: {len(sentiment_cols)}")
        
        return merged
    
    def collect_for_symbol(self, symbol, years=10):
        """
        Collect and merge data for a single symbol
        
        Args:
            symbol: Trading symbol
            years: Years of historical data
            
        Returns:
            Merged DataFrame
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"COLLECTING INTEGRATED DATA FOR {symbol}")
        logger.info(f"{'='*100}")
        
        # Collect price data
        logger.info("\n📊 Step 1: Collecting price data...")
        price_df = self.price_collector.prepare_dataset(symbol, years=years)
        
        if price_df is None or len(price_df) == 0:
            logger.error(f"Failed to collect price data for {symbol}")
            return None
        
        logger.info(f"✅ Price data: {len(price_df)} rows")
        
        # Collect sentiment data
        logger.info("\n📰 Step 2: Collecting sentiment data...")
        
        # Check if sentiment already collected
        sentiment_df = self.sentiment_collector.load_sentiment_data(symbol)
        
        if sentiment_df is None:
            # Collect new sentiment data
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            sentiment_df = self.sentiment_collector.collect_sentiment_for_symbol(
                symbol, start_date, end_date
            )
            self.sentiment_collector.save_sentiment_data(symbol, sentiment_df)
        else:
            logger.info(f"✅ Loaded existing sentiment data: {len(sentiment_df)} days")
        
        # Merge
        logger.info("\n🔗 Step 3: Merging price + sentiment...")
        merged_df = self.merge_price_sentiment(price_df, sentiment_df, symbol)
        
        # Save merged dataset
        output_file = self.output_dir / f'{symbol}_integrated.csv'
        merged_df.to_csv(output_file)
        logger.info(f"✅ Saved integrated dataset: {output_file}")
        
        return merged_df
    
    def collect_all_symbols(self, symbols, years=10):
        """
        Collect and merge data for all symbols
        
        Args:
            symbols: List of trading symbols
            years: Years of historical data
            
        Returns:
            Dictionary of DataFrames
        """
        logger.info("\n" + "="*100)
        logger.info("INTEGRATED DATA COLLECTION")
        logger.info("="*100)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Years: {years}")
        logger.info(f"Components: Price data + News sentiment")
        logger.info("="*100)
        
        results = {}
        
        for symbol in symbols:
            try:
                df = self.collect_for_symbol(symbol, years=years)
                if df is not None:
                    results[symbol] = df
                    logger.info(f"✅ Completed {symbol}")
            except Exception as e:
                logger.error(f"Failed to collect {symbol}: {e}", exc_info=True)
                continue
        
        logger.info("\n" + "="*100)
        logger.info("INTEGRATED DATA COLLECTION COMPLETE")
        logger.info("="*100)
        logger.info(f"Successfully collected: {len(results)} symbols")
        logger.info(f"Saved to: {self.output_dir}/")
        logger.info("="*100)
        
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # NewsAPI key
    NEWSAPI_KEY = '1d76bebe777f4f1b80244b70495b8f16'
    
    # Symbols to collect
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        'BTCUSD', 'ETHUSD',
        'XAUUSD', 'XAGUSD',
        'USOIL', 'UKOIL'
    ]
    
    collector = IntegratedDataCollector(NEWSAPI_KEY)
    
    results = collector.collect_all_symbols(symbols, years=10)
    
    logger.info(f"\n✅ Done! Collected {len(results)} symbols with price + sentiment data")

