#!/usr/bin/env python3
"""
News Sentiment Collector
Collects news from NewsAPI and analyzes sentiment using VADER
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NewsSentimentCollector:
    """Collect and analyze news sentiment for forex symbols"""
    
    def __init__(self, api_key, data_dir='sentiment_data'):
        """
        Initialize news sentiment collector
        
        Args:
            api_key: NewsAPI.org API key
            data_dir: Directory to save sentiment data
        """
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Symbol to currency mapping
        self.symbol_keywords = {
            'EURUSD': ['EUR', 'Euro', 'Dollar', 'USD', 'ECB', 'Federal Reserve'],
            'GBPUSD': ['GBP', 'Pound', 'Sterling', 'Dollar', 'USD', 'Bank of England'],
            'USDJPY': ['USD', 'Dollar', 'JPY', 'Yen', 'Bank of Japan', 'BOJ'],
            'AUDUSD': ['AUD', 'Australian Dollar', 'USD', 'RBA'],
            'USDCAD': ['USD', 'CAD', 'Canadian Dollar', 'Bank of Canada'],
            'NZDUSD': ['NZD', 'New Zealand Dollar', 'USD', 'RBNZ'],
            'EURJPY': ['EUR', 'Euro', 'JPY', 'Yen'],
            'GBPJPY': ['GBP', 'Pound', 'JPY', 'Yen'],
            'AUDJPY': ['AUD', 'Australian Dollar', 'JPY', 'Yen'],
            'BTCUSD': ['Bitcoin', 'BTC', 'cryptocurrency', 'crypto'],
            'ETHUSD': ['Ethereum', 'ETH', 'cryptocurrency', 'crypto'],
            'XAUUSD': ['Gold', 'XAU', 'precious metals'],
            'XAGUSD': ['Silver', 'XAG', 'precious metals'],
            'USOIL': ['Oil', 'Crude', 'WTI', 'petroleum', 'OPEC'],
            'UKOIL': ['Oil', 'Brent', 'petroleum', 'OPEC']
        }
        
    def fetch_news(self, keywords, from_date, to_date):
        """
        Fetch news articles from NewsAPI
        
        Args:
            keywords: List of keywords to search
            from_date: Start date (datetime)
            to_date: End date (datetime)
            
        Returns:
            List of articles
        """
        query = ' OR '.join(keywords[:3])  # Use top 3 keywords
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                return data.get('articles', [])
            else:
                logger.warning(f"API error: {data.get('message', 'Unknown error')}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to +1)
        """
        if not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']  # -1 (negative) to +1 (positive)
    
    def collect_sentiment_for_symbol(self, symbol, start_date, end_date):
        """
        Collect news sentiment for a symbol over a date range
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date (datetime)
            end_date: End date (datetime)
            
        Returns:
            DataFrame with daily sentiment scores
        """
        logger.info(f"Collecting news sentiment for {symbol}...")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        keywords = self.symbol_keywords.get(symbol, [symbol])
        
        # Collect news in monthly chunks (NewsAPI free tier limit)
        all_articles = []
        current_date = start_date
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=30), end_date)
            
            logger.info(f"Fetching news: {current_date.date()} to {chunk_end.date()}")
            articles = self.fetch_news(keywords, current_date, chunk_end)
            all_articles.extend(articles)
            
            logger.info(f"Found {len(articles)} articles")
            
            current_date = chunk_end + timedelta(days=1)
            time.sleep(1)  # Rate limiting
        
        logger.info(f"Total articles collected: {len(all_articles)}")
        
        # Process articles into daily sentiment
        daily_sentiment = {}
        
        for article in all_articles:
            try:
                # Parse date
                pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d')
                date_key = pub_date.date()
                
                # Analyze sentiment
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title}. {description}"
                
                sentiment = self.analyze_sentiment(text)
                
                # Aggregate by date
                if date_key not in daily_sentiment:
                    daily_sentiment[date_key] = {
                        'sentiments': [],
                        'count': 0
                    }
                
                daily_sentiment[date_key]['sentiments'].append(sentiment)
                daily_sentiment[date_key]['count'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process article: {e}")
                continue
        
        # Create DataFrame
        dates = []
        sentiment_scores = []
        news_counts = []
        sentiment_stds = []
        
        current = start_date.date()
        while current <= end_date.date():
            dates.append(current)
            
            if current in daily_sentiment:
                sentiments = daily_sentiment[current]['sentiments']
                sentiment_scores.append(np.mean(sentiments))
                news_counts.append(len(sentiments))
                sentiment_stds.append(np.std(sentiments) if len(sentiments) > 1 else 0)
            else:
                # No news for this day - use neutral
                sentiment_scores.append(0.0)
                news_counts.append(0)
                sentiment_stds.append(0.0)
            
            current += timedelta(days=1)
        
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'news_count': news_counts,
            'sentiment_std': sentiment_stds
        })
        
        # Add rolling features
        df['sentiment_ma_7'] = df['sentiment_score'].rolling(7, min_periods=1).mean()
        df['sentiment_ma_30'] = df['sentiment_score'].rolling(30, min_periods=1).mean()
        df['sentiment_change'] = df['sentiment_score'].diff()
        df['news_volume_ma_7'] = df['news_count'].rolling(7, min_periods=1).mean()
        
        # Major news event flag (high volume or extreme sentiment)
        df['major_news'] = (
            (df['news_count'] > df['news_count'].quantile(0.90)) |
            (df['sentiment_score'].abs() > df['sentiment_score'].abs().quantile(0.90))
        ).astype(int)
        
        logger.info(f"Processed {len(df)} days of sentiment data")
        logger.info(f"Average sentiment: {df['sentiment_score'].mean():.3f}")
        logger.info(f"Average news count: {df['news_count'].mean():.1f} articles/day")
        logger.info(f"Major news days: {df['major_news'].sum()} ({df['major_news'].sum()/len(df)*100:.1f}%)")
        
        return df
    
    def save_sentiment_data(self, symbol, df):
        """Save sentiment data to CSV"""
        file_path = self.data_dir / f'{symbol}_sentiment.csv'
        df.to_csv(file_path, index=False)
        logger.info(f"✅ Saved sentiment data: {file_path}")
    
    def load_sentiment_data(self, symbol):
        """Load sentiment data from CSV"""
        file_path = self.data_dir / f'{symbol}_sentiment.csv'
        if file_path.exists():
            return pd.read_csv(file_path, parse_dates=['date'])
        return None
    
    def collect_all_symbols(self, symbols, years=10):
        """
        Collect sentiment data for all symbols
        
        Args:
            symbols: List of trading symbols
            years: Number of years of historical data
            
        Returns:
            Dictionary of DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        results = {}
        
        for symbol in symbols:
            try:
                df = self.collect_sentiment_for_symbol(symbol, start_date, end_date)
                self.save_sentiment_data(symbol, df)
                results[symbol] = df
                
                logger.info(f"✅ Completed {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to collect sentiment for {symbol}: {e}", exc_info=True)
                continue
            
            # Rate limiting (NewsAPI free tier: 100 requests/day)
            time.sleep(2)
        
        logger.info(f"\n✅ Sentiment collection complete for {len(results)} symbols")
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # NewsAPI key
    API_KEY = '1d76bebe777f4f1b80244b70495b8f16'
    
    # Symbols to collect
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD',
        'EURJPY', 'GBPJPY', 'AUDJPY',
        'BTCUSD', 'ETHUSD',
        'XAUUSD', 'XAGUSD',
        'USOIL', 'UKOIL'
    ]
    
    collector = NewsSentimentCollector(API_KEY)
    
    logger.info("\n" + "="*100)
    logger.info("NEWS SENTIMENT COLLECTOR")
    logger.info("="*100)
    logger.info(f"Collecting sentiment for {len(symbols)} symbols")
    logger.info(f"Date range: Last 10 years")
    logger.info(f"Source: NewsAPI.org")
    logger.info(f"Analyzer: VADER Sentiment")
    logger.info("="*100)
    
    results = collector.collect_all_symbols(symbols, years=10)
    
    logger.info("\n" + "="*100)
    logger.info("SENTIMENT COLLECTION COMPLETE")
    logger.info("="*100)
    logger.info(f"Successfully collected: {len(results)} symbols")
    logger.info(f"Saved to: sentiment_data/")
    logger.info("="*100)

