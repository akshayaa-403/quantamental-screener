"""
News sentiment analysis using FinBERT, VADER, and TextBlob.
Fetches headlines from Yahoo Finance (with fallback to sample data).
"""
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config.settings import Config


class SentimentAnalyzer:
    """Analyzes news sentiment for stocks"""

    def __init__(self, config: Config):
        self.config = config
        self.sentiment_data = {}

        # Initialize sentiment analyzers
        print("Loading sentiment analysis models...")
        try:
            self.finbert = pipeline("sentiment-analysis",
                                    model="ProsusAI/finbert",
                                    tokenizer="ProsusAI/finbert")
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"FinBERT not available: {e}")
            self.finbert = None

        self.vader = SentimentIntensityAnalyzer()
        print("VADER sentiment analyzer loaded")

    def get_yahoo_news(self, ticker: str) -> List[Dict]:
        """Scrape news from Yahoo Finance"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = []
            news_items = soup.find_all('div', class_='Ov(h)')[:self.config.MAX_NEWS_PER_STOCK]

            for item in news_items:
                try:
                    headline = item.find('h3')
                    if headline:
                        articles.append({
                            'headline': headline.get_text().strip(),
                            'source': 'yahoo',
                            'date': datetime.now().isoformat(),
                            'ticker': ticker
                        })
                except:
                    continue

            return articles

        except Exception as e:
            print(f"Error fetching Yahoo news for {ticker}: {e}")
            return []

    def get_sample_news(self, ticker: str) -> List[Dict]:
        """Generate sample news for demonstration (fallback)"""
        sample_headlines = [
            f"{ticker} reports strong quarterly earnings",
            f"{ticker} announces new product launch",
            f"Analysts upgrade {ticker} stock rating",
            f"{ticker} faces regulatory challenges",
            f"{ticker} stock shows technical breakout"
        ]

        return [{'headline': headline, 'source': 'sample', 'date': datetime.now().isoformat(), 'ticker': ticker}
                for headline in sample_headlines[:2]]

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using multiple methods"""
        results = {}

        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        results['vader'] = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu']
        }

        # FinBERT sentiment (if available)
        if self.finbert:
            try:
                finbert_result = self.finbert(text[:512])  # FinBERT has token limit
                results['finbert'] = {
                    'label': finbert_result[0]['label'],
                    'score': finbert_result[0]['score']
                }
            except Exception:
                results['finbert'] = {'label': 'NEUTRAL', 'score': 0.5}

        # TextBlob sentiment
        blob = TextBlob(text)
        results['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

        return results

    def calculate_composite_sentiment(self, sentiment_results: Dict) -> float:
        """Calculate composite sentiment score from multiple analyzers"""
        scores = []

        # VADER compound score
        if 'vader' in sentiment_results:
            scores.append(sentiment_results['vader']['compound'])

        # FinBERT score (convert to -1 to 1 scale)
        if 'finbert' in sentiment_results:
            finbert_score = sentiment_results['finbert']['score']
            if sentiment_results['finbert']['label'] == 'negative':
                finbert_score = -finbert_score
            elif sentiment_results['finbert']['label'] == 'neutral':
                finbert_score = 0
            scores.append(finbert_score)

        # TextBlob polarity
        if 'textblob' in sentiment_results:
            scores.append(sentiment_results['textblob']['polarity'])

        return np.mean(scores) if scores else 0.0

    def analyze_stock_sentiment(self, ticker: str) -> Dict:
        """Analyze sentiment for a single stock"""
        # Get news articles
        articles = self.get_yahoo_news(ticker)
        if not articles:
            articles = self.get_sample_news(ticker)

        if not articles:
            return {
                'overall_sentiment': 0.0,
                'article_count': 0,
                'sentiment_scores': [],
                'articles': []
            }

        # Analyze each article
        sentiment_scores = []
        analyzed_articles = []

        for article in articles:
            sentiment_result = self.analyze_sentiment(article['headline'])
            composite_score = self.calculate_composite_sentiment(sentiment_result)

            sentiment_scores.append(composite_score)
            analyzed_articles.append({
                **article,
                'sentiment_score': composite_score,
                'sentiment_detail': sentiment_result
            })

        # Calculate overall sentiment
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

        return {
            'overall_sentiment': overall_sentiment,
            'article_count': len(articles),
            'sentiment_scores': sentiment_scores,
            'articles': analyzed_articles,
            'sentiment_std': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        }

    def analyze_all_stocks(self, tickers: List[str]) -> Dict[str, Dict]:
        """Analyze sentiment for all stocks"""
        print(f"Analyzing sentiment for {len(tickers)} stocks...")

        for i, ticker in enumerate(tickers):
            if i % 5 == 0:
                print(f"   Analyzed {i}/{len(tickers)}")

            try:
                sentiment_result = self.analyze_stock_sentiment(ticker)
                self.sentiment_data[ticker] = sentiment_result
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error analyzing sentiment for {ticker}: {e}")
                continue

        print(f"Sentiment analysis completed for {len(self.sentiment_data)} stocks")
        return self.sentiment_data