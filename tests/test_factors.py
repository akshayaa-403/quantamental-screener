"""
Unit tests for all factor implementations.
"""

import pytest
import pandas as pd
import numpy as np
from factors import MomentumFactor, SentimentFactor, VolumeFactor, VolatilityFactor


class TestMomentumFactor:
    """Test MomentumFactor calculations."""
    
    def setup_method(self):
        self.factor = MomentumFactor()
        self.mock_data = self._create_mock_price_data()
    
    def test_compute_returns_series(self):
        """Test that compute returns a pandas Series."""
        result = self.factor.compute(self.mock_data)
        assert isinstance(result, pd.Series)
        assert len(result) == 2  # AAPL, MSFT
    
    def test_normalize(self):
        """Test normalization function."""
        scores = pd.Series([-0.5, 0, 0.5])
        normalized = self.factor.normalize(scores)
        assert normalized.min() >= -1
        assert normalized.max() <= 1
    
    def _create_mock_price_data(self):
        """Create mock price data with sufficient history."""
        dates = pd.date_range('2024-01-01', '2024-02-01', freq='D')
        tickers = ['AAPL', 'MSFT']
        
        data = []
        for ticker in tickers:
            base_price = 150 if ticker == 'AAPL' else 300
            for i, date in enumerate(dates):
                data.append({
                    'Ticker': ticker,
                    'Date': date,
                    'Open': base_price + i,
                    'High': base_price + i + 1,
                    'Low': base_price + i - 1,
                    'Close': base_price + i,
                    'Volume': 1000000
                })
        
        df = pd.DataFrame(data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        return df


class TestSentimentFactor:
    """Test SentimentFactor."""
    
    def test_compute_with_sentiment_scores(self):
        """Test that sentiment factor uses passed sentiment scores."""
        factor = SentimentFactor()
        mock_data = pd.DataFrame()
        sentiment_scores = {'AAPL': {'score': 0.8}, 'MSFT': {'score': -0.2}}
        
        result = factor.compute(mock_data, sentiment_scores=sentiment_scores)
        
        assert result['AAPL'] == 0.8
        assert result['MSFT'] == -0.2
    
    def test_compute_without_sentiment(self):
        """Test default behavior when no sentiment scores provided."""
        factor = SentimentFactor()
        mock_data = pd.DataFrame(index=pd.MultiIndex.from_tuples([('AAPL', '2024-01-01')], names=['Ticker', 'Date']))
        
        result = factor.compute(mock_data)
        assert result['AAPL'] == 0.0


class TestVolumeFactor:
    """Test VolumeFactor calculations."""
    
    def test_compute_volume_ratio(self):
        """Test volume ratio calculation."""
        factor = VolumeFactor()
        mock_data = self._create_mock_volume_data()
        result = factor.compute(mock_data)
        
        assert isinstance(result, pd.Series)
        assert result.index[0] in ['AAPL', 'MSFT']
    
    def _create_mock_volume_data(self):
        """Create mock data with varying volumes."""
        dates = pd.date_range('2024-01-01', '2024-01-30', freq='D')
        tickers = ['AAPL']
        
        data = []
        for i, date in enumerate(dates):
            # Increasing volume over time
            volume = 1000000 + (i * 50000)
            data.append({
                'Ticker': 'AAPL',
                'Date': date,
                'Open': 150,
                'High': 151,
                'Low': 149,
                'Close': 150,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        return df


class TestVolatilityFactor:
    """Test VolatilityFactor calculations."""
    
    def test_higher_volatility_yields_lower_score(self):
        """Test that higher volatility produces lower (worse) scores."""
        factor = VolatilityFactor()
        
        dates = pd.date_range('2024-01-01', '2024-02-01', freq='D')
        
        data = []
        for date in dates:
            data.append({
                'Ticker': 'LOW_VOL',
                'Date': date,
                'Open': 100,
                'High': 101,
                'Low': 99,
                'Close': 100 + np.random.normal(0, 0.5),
                'Volume': 1000000
            })
        
        for date in dates:
            data.append({
                'Ticker': 'HIGH_VOL',
                'Date': date,
                'Open': 100,
                'High': 105,
                'Low': 95,
                'Close': 100 + np.random.normal(0, 3),
                'Volume': 1000000
            })
        
        df = pd.DataFrame(data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        
        result = factor.compute(df)
        
        assert result['LOW_VOL'] > result['HIGH_VOL']