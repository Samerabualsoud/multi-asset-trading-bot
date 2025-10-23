#!/usr/bin/env python3
"""
LLM News Analyst
Uses LLM (DeepSeek/ChatGPT) to analyze news articles for trading impact
Much more sophisticated than VADER sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import time
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LLMNewsAnalyst:
    """Analyze news articles using LLM for trading insights"""
    
    def __init__(self, api_key, model='deepseek-chat', api_type='deepseek'):
        """
        Initialize LLM news analyst
        
        Args:
            api_key: API key (DeepSeek or OpenAI)
            model: Model name ('deepseek-chat' or 'gpt-4o-mini')
            api_type: 'deepseek' or 'openai'
        """
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        
        # API endpoints
        if api_type == 'deepseek':
            self.api_url = 'https://api.deepseek.com/v1/chat/completions'
        else:  # openai
            self.api_url = 'https://api.openai.com/v1/chat/completions'
        
        # Symbol to currency mapping
        self.symbol_info = {
            'EURUSD': {'base': 'EUR', 'quote': 'USD', 'type': 'forex'},
            'GBPUSD': {'base': 'GBP', 'quote': 'USD', 'type': 'forex'},
            'USDJPY': {'base': 'USD', 'quote': 'JPY', 'type': 'forex'},
            'AUDUSD': {'base': 'AUD', 'quote': 'USD', 'type': 'forex'},
            'USDCAD': {'base': 'USD', 'quote': 'CAD', 'type': 'forex'},
            'NZDUSD': {'base': 'NZD', 'quote': 'USD', 'type': 'forex'},
            'EURJPY': {'base': 'EUR', 'quote': 'JPY', 'type': 'forex'},
            'GBPJPY': {'base': 'GBP', 'quote': 'JPY', 'type': 'forex'},
            'AUDJPY': {'base': 'AUD', 'quote': 'JPY', 'type': 'forex'},
            'BTCUSD': {'base': 'BTC', 'quote': 'USD', 'type': 'crypto'},
            'ETHUSD': {'base': 'ETH', 'quote': 'USD', 'type': 'crypto'},
            'XAUUSD': {'base': 'Gold', 'quote': 'USD', 'type': 'commodity'},
            'XAGUSD': {'base': 'Silver', 'quote': 'USD', 'type': 'commodity'},
            'USOIL': {'base': 'Oil', 'quote': 'USD', 'type': 'commodity'},
            'UKOIL': {'base': 'Oil', 'quote': 'USD', 'type': 'commodity'}
        }
    
    def analyze_news_article(self, symbol, title, description, date):
        """
        Analyze a single news article using LLM
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            title: News title
            description: News description
            date: Publication date
            
        Returns:
            Dictionary with analysis results
        """
        info = self.symbol_info.get(symbol, {})
        base = info.get('base', symbol[:3])
        quote = info.get('quote', symbol[3:])
        asset_type = info.get('type', 'forex')
        
        # Construct prompt
        prompt = f"""You are a professional financial analyst. Analyze this news article for its impact on {symbol} ({base}/{quote}).

News Date: {date}
Title: {title}
Description: {description}

Provide analysis in this exact JSON format:
{{
    "impact_direction": "bullish" or "bearish" or "neutral",
    "impact_score": <number from -1.0 (very bearish) to +1.0 (very bullish)>,
    "confidence": <number from 0 to 100>,
    "timeframe": "immediate" or "short-term" or "medium-term" or "long-term",
    "reasoning": "<brief explanation in 1-2 sentences>",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Consider:
- Interest rate changes, economic data, central bank policy
- Geopolitical events, trade relations
- Market sentiment, risk appetite
- Supply/demand factors for commodities
- Adoption/regulation for crypto

Respond ONLY with valid JSON, no other text."""

        try:
            # Call LLM API
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': 'You are a professional financial analyst providing structured analysis.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,  # Lower temperature for more consistent analysis
                'max_tokens': 500
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            analysis = json.loads(content)
            
            # Validate and normalize
            analysis['impact_score'] = max(-1.0, min(1.0, float(analysis.get('impact_score', 0))))
            analysis['confidence'] = max(0, min(100, int(analysis.get('confidence', 50))))
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.warning(f"Response content: {content[:200]}")
            return self._fallback_analysis()
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self):
        """Return neutral analysis if LLM fails"""
        return {
            'impact_direction': 'neutral',
            'impact_score': 0.0,
            'confidence': 0,
            'timeframe': 'unknown',
            'reasoning': 'Analysis failed, defaulting to neutral',
            'key_factors': []
        }
    
    def analyze_news_batch(self, symbol, articles, batch_size=10):
        """
        Analyze multiple news articles for a symbol
        
        Args:
            symbol: Trading symbol
            articles: List of articles (each with title, description, date)
            batch_size: Number of articles to process (rate limiting)
            
        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing {min(len(articles), batch_size)} articles for {symbol} with LLM...")
        
        results = []
        
        for i, article in enumerate(articles[:batch_size]):
            try:
                title = article.get('title', '')
                description = article.get('description', '')
                date = article.get('publishedAt', '')[:10]
                
                if not title and not description:
                    continue
                
                analysis = self.analyze_news_article(symbol, title, description, date)
                analysis['title'] = title
                analysis['date'] = date
                
                results.append(analysis)
                
                # Rate limiting
                time.sleep(0.5)  # 2 requests/second max
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{min(len(articles), batch_size)} articles")
                
            except Exception as e:
                logger.error(f"Failed to analyze article: {e}")
                continue
        
        logger.info(f"✅ Analyzed {len(results)} articles")
        
        return results
    
    def aggregate_daily_sentiment(self, analyses):
        """
        Aggregate LLM analyses into daily sentiment scores
        
        Args:
            analyses: List of LLM analysis results
            
        Returns:
            DataFrame with daily sentiment
        """
        if not analyses:
            return pd.DataFrame()
        
        # Group by date
        daily_data = {}
        
        for analysis in analyses:
            date = analysis['date']
            
            if date not in daily_data:
                daily_data[date] = {
                    'scores': [],
                    'confidences': [],
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'timeframes': [],
                    'key_factors': []
                }
            
            # Aggregate
            score = analysis['impact_score']
            confidence = analysis['confidence']
            direction = analysis['impact_direction']
            
            daily_data[date]['scores'].append(score)
            daily_data[date]['confidences'].append(confidence)
            
            if direction == 'bullish':
                daily_data[date]['bullish_count'] += 1
            elif direction == 'bearish':
                daily_data[date]['bearish_count'] += 1
            else:
                daily_data[date]['neutral_count'] += 1
            
            daily_data[date]['timeframes'].append(analysis.get('timeframe', 'unknown'))
            daily_data[date]['key_factors'].extend(analysis.get('key_factors', []))
        
        # Create DataFrame
        rows = []
        for date, data in daily_data.items():
            # Weighted average (by confidence)
            scores = np.array(data['scores'])
            confidences = np.array(data['confidences'])
            
            if len(confidences) > 0 and confidences.sum() > 0:
                weighted_score = np.average(scores, weights=confidences)
            else:
                weighted_score = np.mean(scores) if len(scores) > 0 else 0.0
            
            rows.append({
                'date': pd.to_datetime(date),
                'llm_sentiment_score': weighted_score,
                'llm_confidence': np.mean(confidences) if len(confidences) > 0 else 0,
                'news_count': len(data['scores']),
                'bullish_count': data['bullish_count'],
                'bearish_count': data['bearish_count'],
                'neutral_count': data['neutral_count'],
                'sentiment_std': np.std(scores) if len(scores) > 1 else 0
            })
        
        df = pd.DataFrame(rows).sort_values('date')
        
        # Add rolling features
        df['llm_sentiment_ma_7'] = df['llm_sentiment_score'].rolling(7, min_periods=1).mean()
        df['llm_sentiment_ma_30'] = df['llm_sentiment_score'].rolling(30, min_periods=1).mean()
        df['llm_sentiment_change'] = df['llm_sentiment_score'].diff()
        df['llm_confidence_ma_7'] = df['llm_confidence'].rolling(7, min_periods=1).mean()
        
        # Major news flag (high confidence + extreme sentiment)
        df['llm_major_news'] = (
            (df['llm_confidence'] > 70) &
            (df['llm_sentiment_score'].abs() > 0.5)
        ).astype(int)
        
        return df
    
    def save_analyses(self, symbol, analyses, output_dir='llm_analyses'):
        """Save LLM analyses to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / f'{symbol}_llm_analyses.json'
        with open(file_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        logger.info(f"✅ Saved LLM analyses: {file_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with DeepSeek
    DEEPSEEK_API_KEY = 'sk-beecc0804a1546558463cff42e14d694'
    
    analyst = LLMNewsAnalyst(DEEPSEEK_API_KEY, model='deepseek-chat', api_type='deepseek')
    
    # Test analysis
    test_article = {
        'title': 'Federal Reserve raises interest rates by 0.5%',
        'description': 'The Fed announced a 50 basis point rate hike to combat inflation, signaling more increases ahead.',
        'publishedAt': '2024-03-15'
    }
    
    logger.info("Testing LLM News Analyst...")
    analysis = analyst.analyze_news_article(
        'EURUSD',
        test_article['title'],
        test_article['description'],
        test_article['publishedAt']
    )
    
    logger.info(f"\nAnalysis Result:")
    logger.info(json.dumps(analysis, indent=2))

