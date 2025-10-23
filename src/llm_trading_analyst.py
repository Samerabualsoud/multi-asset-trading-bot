#!/usr/bin/env python3
"""
LLM Trading Analyst
Final decision layer that reviews ML predictions and all available data
Acts as a professional trading analyst providing final trade recommendations
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LLMTradingAnalyst:
    """LLM-powered trading analyst for final trade decisions"""
    
    def __init__(self, api_key, model='deepseek-chat', api_type='deepseek'):
        """
        Initialize LLM trading analyst
        
        Args:
            api_key: API key (DeepSeek or OpenAI)
            model: Model name
            api_type: 'deepseek' or 'openai'
        """
        self.api_key = api_key
        self.model = model
        self.api_type = api_type
        
        if api_type == 'deepseek':
            self.api_url = 'https://api.deepseek.com/v1/chat/completions'
        else:
            self.api_url = 'https://api.openai.com/v1/chat/completions'
    
    def analyze_trade_opportunity(self, symbol, ml_prediction, ml_confidence, 
                                  technical_data, sentiment_data):
        """
        Analyze a trade opportunity using LLM
        
        Args:
            symbol: Trading symbol
            ml_prediction: ML model prediction ('BUY' or 'SELL')
            ml_confidence: ML model confidence (0-100)
            technical_data: Dict with technical indicators
            sentiment_data: Dict with sentiment analysis
            
        Returns:
            Dictionary with LLM analysis and final decision
        """
        
        # Construct comprehensive prompt
        prompt = f"""You are a professional forex/crypto trader. Analyze this trade opportunity for {symbol}.

**ML Model Prediction:**
- Signal: {ml_prediction}
- Confidence: {ml_confidence}%

**Technical Analysis:**
- Trend: {technical_data.get('trend', 'unknown')}
- RSI: {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd_signal', 'unknown')}
- Moving Averages: {technical_data.get('ma_alignment', 'unknown')}
- Volatility: {technical_data.get('volatility', 'N/A')}
- ADX: {technical_data.get('adx', 'N/A')}

**Sentiment Analysis:**
- News Sentiment: {sentiment_data.get('sentiment_score', 0):.2f} (-1 to +1)
- Confidence: {sentiment_data.get('confidence', 0):.0f}%
- Recent News: {sentiment_data.get('news_count', 0)} articles
- Major News: {'Yes' if sentiment_data.get('major_news', 0) else 'No'}

**Task:**
Review all data and provide your professional analysis in this exact JSON format:
{{
    "final_decision": "BUY" or "SELL" or "HOLD",
    "confidence": <0-100>,
    "reasoning": "<2-3 sentences explaining your decision>",
    "risk_level": "low" or "medium" or "high",
    "key_factors": ["factor1", "factor2", "factor3"],
    "concerns": ["concern1", "concern2"] or [],
    "suggested_tp_multiplier": <0.5 to 2.0>,
    "trade_quality": "excellent" or "good" or "fair" or "poor"
}}

Consider:
1. Does ML prediction align with technical indicators?
2. Does sentiment support the trade direction?
3. Is trend strength sufficient?
4. Are there conflicting signals?
5. What is the risk/reward ratio?

Respond ONLY with valid JSON, no other text."""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': 'You are a professional trader providing structured trade analysis.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.2,  # Low temperature for consistent decisions
                'max_tokens': 600
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
            
            # Parse JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            analysis = json.loads(content)
            
            # Validate
            analysis['confidence'] = max(0, min(100, int(analysis.get('confidence', 50))))
            analysis['suggested_tp_multiplier'] = max(0.5, min(2.0, float(analysis.get('suggested_tp_multiplier', 1.0))))
            
            # Add metadata
            analysis['ml_prediction'] = ml_prediction
            analysis['ml_confidence'] = ml_confidence
            analysis['symbol'] = symbol
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_decision(ml_prediction, ml_confidence)
            
        except Exception as e:
            logger.error(f"LLM trading analysis failed: {e}")
            return self._fallback_decision(ml_prediction, ml_confidence)
    
    def _fallback_decision(self, ml_prediction, ml_confidence):
        """Fallback to ML prediction if LLM fails"""
        return {
            'final_decision': ml_prediction if ml_confidence > 70 else 'HOLD',
            'confidence': max(0, ml_confidence - 10),  # Reduce confidence slightly
            'reasoning': 'LLM analysis unavailable, using ML prediction with reduced confidence',
            'risk_level': 'medium',
            'key_factors': ['ML prediction only'],
            'concerns': ['LLM analysis failed'],
            'suggested_tp_multiplier': 1.0,
            'trade_quality': 'fair',
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence
        }
    
    def batch_analyze(self, opportunities):
        """
        Analyze multiple trade opportunities
        
        Args:
            opportunities: List of dicts with trade data
            
        Returns:
            List of LLM analyses
        """
        results = []
        
        for opp in opportunities:
            try:
                analysis = self.analyze_trade_opportunity(
                    opp['symbol'],
                    opp['ml_prediction'],
                    opp['ml_confidence'],
                    opp['technical_data'],
                    opp['sentiment_data']
                )
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze {opp.get('symbol', 'unknown')}: {e}")
                continue
        
        return results
    
    def get_trade_summary(self, analysis):
        """
        Generate human-readable trade summary
        
        Args:
            analysis: LLM analysis dict
            
        Returns:
            Formatted string
        """
        summary = f"""
{'='*80}
TRADE ANALYSIS: {analysis.get('symbol', 'N/A')}
{'='*80}

ML Prediction: {analysis.get('ml_prediction', 'N/A')} ({analysis.get('ml_confidence', 0)}%)
LLM Decision: {analysis.get('final_decision', 'N/A')} ({analysis.get('confidence', 0)}%)

Quality: {analysis.get('trade_quality', 'N/A').upper()}
Risk Level: {analysis.get('risk_level', 'N/A').upper()}

Reasoning:
{analysis.get('reasoning', 'N/A')}

Key Factors:
{chr(10).join('  • ' + f for f in analysis.get('key_factors', []))}

{f"Concerns:{chr(10)}{chr(10).join('  ⚠️ ' + c for c in analysis.get('concerns', []))}" if analysis.get('concerns') else "No major concerns"}

Suggested TP Multiplier: {analysis.get('suggested_tp_multiplier', 1.0):.1f}x
{'='*80}
"""
        return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with DeepSeek
    DEEPSEEK_API_KEY = 'sk-beecc0804a1546558463cff42e14d694'
    
    analyst = LLMTradingAnalyst(DEEPSEEK_API_KEY, model='deepseek-chat', api_type='deepseek')
    
    # Test analysis
    test_opportunity = {
        'symbol': 'EURUSD',
        'ml_prediction': 'BUY',
        'ml_confidence': 75,
        'technical_data': {
            'trend': 'uptrend',
            'rsi': 58.5,
            'macd_signal': 'bullish',
            'ma_alignment': 'bullish (EMA20 > MA20 > MA50)',
            'volatility': 'moderate',
            'adx': 28.5
        },
        'sentiment_data': {
            'sentiment_score': 0.65,
            'confidence': 80,
            'news_count': 12,
            'major_news': 1
        }
    }
    
    logger.info("Testing LLM Trading Analyst...")
    analysis = analyst.analyze_trade_opportunity(
        test_opportunity['symbol'],
        test_opportunity['ml_prediction'],
        test_opportunity['ml_confidence'],
        test_opportunity['technical_data'],
        test_opportunity['sentiment_data']
    )
    
    logger.info("\n" + analyst.get_trade_summary(analysis))

