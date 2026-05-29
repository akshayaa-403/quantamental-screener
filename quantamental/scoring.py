"""
Scoring engine – computes momentum, sentiment, volume, and volatility scores,
then combines them into a composite score with configurable weights.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from config.settings import Config


class ScoringEngine:
    """Calculates composite scores and ranks stocks"""

    def __init__(self, config: Config, data_collector, sentiment_data: Dict, universe_selector):
        self.config = config
        self.data_collector = data_collector
        self.sentiment_data = sentiment_data
        self.universe_selector = universe_selector
        self.scores = {}
        self.rankings = {}

    def calculate_momentum_score(self, ticker: str, latest_values: Dict) -> float:
        """Calculate momentum score from technical indicators"""
        try:
            score = 0.0
            components = 0

            # RSI momentum (inverse for mean reversion)
            if 'rsi' in latest_values and not pd.isna(latest_values['rsi']):
                rsi = latest_values['rsi']
                if rsi > 70:
                    score += -0.3  # Overbought
                elif rsi < 30:
                    score += 0.3   # Oversold
                else:
                    score += (50 - rsi) / 100
                components += 1

            # MACD momentum
            if 'macd' in latest_values and 'macd_signal' in latest_values:
                macd = latest_values['macd']
                macd_signal = latest_values['macd_signal']
                if not (pd.isna(macd) or pd.isna(macd_signal)):
                    macd_diff = macd - macd_signal
                    score += np.tanh(macd_diff) * 0.3
                    components += 1

            # Moving average crossover
            if 'sma_20' in latest_values and 'sma_50' in latest_values:
                sma_20 = latest_values['sma_20']
                sma_50 = latest_values['sma_50']
                if not (pd.isna(sma_20) or pd.isna(sma_50)):
                    ma_signal = (sma_20 - sma_50) / sma_50
                    score += np.tanh(ma_signal * 10) * 0.2
                    components += 1

            # ADX trend strength
            if 'adx' in latest_values and not pd.isna(latest_values['adx']):
                adx = latest_values['adx']
                if adx > 25:
                    score += 0.2
                components += 1

            return score / components if components > 0 else 0.0

        except Exception as e:
            print(f"Error calculating momentum score for {ticker}: {e}")
            return 0.0

    def calculate_sentiment_score(self, ticker: str) -> float:
        """Calculate sentiment score"""
        try:
            if ticker not in self.sentiment_data:
                return 0.0

            sentiment_info = self.sentiment_data[ticker]
            base_score = sentiment_info['overall_sentiment']

            # Weight by number of articles (more articles = more confidence)
            article_count = sentiment_info['article_count']
            confidence_weight = min(1.0, article_count / 5.0)

            # Penalize high sentiment volatility (inconsistent news)
            sentiment_std = sentiment_info.get('sentiment_std', 0.0)
            volatility_penalty = min(0.3, sentiment_std)

            final_score = base_score * confidence_weight - volatility_penalty
            return np.clip(final_score, -1.0, 1.0)

        except Exception as e:
            print(f"Error calculating sentiment score for {ticker}: {e}")
            return 0.0

    def calculate_volume_score(self, ticker: str, latest_values: Dict) -> float:
        """Calculate volume-based score"""
        try:
            score = 0.0

            # Chaikin Money Flow
            if 'cmf' in latest_values:
                cmf = latest_values['cmf']
                if isinstance(cmf, pd.Series):
                    cmf = cmf.iloc[-1]
                if not pd.isna(cmf):
                    score += cmf * 0.5

            # Volume trend (compare current volume to average)
            if 'volume' in latest_values and 'volume_sma' in latest_values:
                volume = latest_values['volume']
                volume_sma = latest_values['volume_sma']

                if isinstance(volume, pd.Series):
                    volume = volume.iloc[-1]
                if isinstance(volume_sma, pd.Series):
                    volume_sma = volume_sma.iloc[-1]

                if not (pd.isna(volume) or pd.isna(volume_sma)) and volume_sma > 0:
                    volume_ratio = volume / volume_sma
                    score += np.tanh(np.log(volume_ratio)) * 0.3

            return np.clip(score, -1.0, 1.0)

        except Exception as e:
            print(f"Error calculating volume score for {ticker}: {e}")
            return 0.0

    def calculate_volatility_score(self, ticker: str, latest_values: Dict) -> float:
        """Calculate volatility-based score (risk adjustment)"""
        try:
            score = 0.0

            # Bollinger Band width (volatility measure)
            if 'bb_width' in latest_values:
                bb_width = latest_values['bb_width']
                if isinstance(bb_width, pd.Series):
                    bb_width = bb_width.iloc[-1]
                if not pd.isna(bb_width):
                    score -= min(0.5, bb_width * 2)

            # ATR relative to price
            if 'atr' in latest_values and 'price' in latest_values:
                atr = latest_values['atr']
                price = latest_values['price']
                if isinstance(atr, pd.Series):
                    atr = atr.iloc[-1]
                if isinstance(price, pd.Series):
                    price = price.iloc[-1]
                if not (pd.isna(atr) or pd.isna(price)) and price > 0:
                    relative_atr = atr / price
                    score -= min(0.3, relative_atr * 10)

            return np.clip(score, -1.0, 1.0)

        except Exception as e:
            print(f"Error calculating volatility score for {ticker}: {e}")
            return 0.0

    def calculate_composite_score(self, ticker: str) -> Dict:
        """Calculate composite score for a stock"""
        try:
            latest_values = self.data_collector.get_latest_values(ticker)

            if not latest_values:
                return {
                    'ticker': ticker,
                    'composite_score': 0.0,
                    'momentum_score': 0.0,
                    'sentiment_score': 0.0,
                    'volume_score': 0.0,
                    'volatility_score': 0.0,
                    'error': 'No data available'
                }

            momentum_score = self.calculate_momentum_score(ticker, latest_values)
            sentiment_score = self.calculate_sentiment_score(ticker)
            volume_score = self.calculate_volume_score(ticker, latest_values)
            volatility_score = self.calculate_volatility_score(ticker, latest_values)

            composite_score = (
                momentum_score * self.config.MOMENTUM_WEIGHT +
                sentiment_score * self.config.SENTIMENT_WEIGHT +
                volume_score * self.config.VOLUME_WEIGHT +
                volatility_score * self.config.VOLATILITY_WEIGHT
            )

            return {
                'ticker': ticker,
                'composite_score': composite_score,
                'momentum_score': momentum_score,
                'sentiment_score': sentiment_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'price': latest_values.get('price', 0),
                'volume': latest_values.get('volume', 0),
                'sector': self.universe_selector.get_sector_info(ticker).get('sector', 'Unknown')
            }

        except Exception as e:
            print(f"Error calculating composite score for {ticker}: {e}")
            return {
                'ticker': ticker,
                'composite_score': 0.0,
                'momentum_score': 0.0,
                'sentiment_score': 0.0,
                'volume_score': 0.0,
                'volatility_score': 0.0,
                'error': str(e)
            }

    def score_all_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """Score all stocks and return ranked DataFrame"""
        print(f"Scoring {len(tickers)} stocks...")

        all_scores = []
        for i, ticker in enumerate(tickers):
            if i % 10 == 0:
                print(f"   Scored {i}/{len(tickers)}")

            score_data = self.calculate_composite_score(ticker)
            all_scores.append(score_data)

        scores_df = pd.DataFrame(all_scores)
        scores_df = scores_df.sort_values('composite_score', ascending=False)
        scores_df['rank'] = range(1, len(scores_df) + 1)

        print(f"Scoring completed for {len(scores_df)} stocks")
        return scores_df

    def get_top_stocks(self, scores_df: pd.DataFrame, n: int = None) -> pd.DataFrame:
        """Get top N stocks"""
        if n is None:
            n = self.config.TOP_N_STOCKS
        return scores_df.head(n)