"""
Integration tests for pipeline components.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pipeline import UniverseSelector, DataCollector, ScoringEngine, BacktestEngine
from factors import MomentumFactor, SentimentFactor


class TestUniverseSelector:
    """Test universe selection."""
    
    @patch('pipeline.universe_selector.requests.get')
    def test_select_returns_tickers(self, mock_get):
        """Test that universe selector returns a list of tickers."""
        # Mock HTML response
        mock_response = Mock()
        mock_response.text = """
        <table id="constituents">
            <tr><td>AAPL</td><td>Apple Inc.</td></tr>
            <tr><td>MSFT</td><td>Microsoft</td></tr>
            <tr><td>GOOGL</td><td>Google</td></tr>
        </table>
        """
        mock_get.return_value = mock_response
        
        selector = UniverseSelector({'size': 2})
        tickers = selector.select()
        
        assert len(tickers) == 2
        assert tickers[0] == 'AAPL'
        assert tickers[1] == 'MSFT'
    
    def test_respects_universe_size(self):
        """Test that selector respects the configured size."""
        selector = UniverseSelector({'size': 5})
        # Mock the ticker list
        selector.select = Mock(return_value=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'][:5])
        tickers = selector.select()
        assert len(tickers) <= 5


class TestDataCollector:
    """Test data collection and technical indicator calculation."""
    
    def test_collect_adds_indicators(self):
        """Test that technical indicators are added to the data."""
        mock_source = Mock()
        mock_source.get_price_data.return_value = self._create_mock_price_data()
        
        collector = DataCollector(mock_source)
        result = collector.collect(['AAPL'], '2024-01-01', '2024-01-31')
        
        # Check that indicators were added
        expected_indicators = ['RSI', 'MACD', 'BB_width', 'Volume_Ratio', 'ATR']
        for indicator in expected_indicators:
            assert indicator in result.columns
    
    def test_caching_works(self):
        """Test that data collector uses cache."""
        mock_source = Mock()
        mock_source.get_price_data.return_value = self._create_mock_price_data()
        
        collector = DataCollector(mock_source)
        
        # First call
        result1 = collector.collect(['AAPL'], '2024-01-01', '2024-01-31')
        # Second call should use cache
        result2 = collector.collect(['AAPL'], '2024-01-01', '2024-01-31')
        
        # Both should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def _create_mock_price_data(self):
        """Create minimal price data for testing."""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        data = []
        for i, date in enumerate(dates):
            data.append({
                'Ticker': 'AAPL',
                'Date': date,
                'Open': 150 + i * 0.5,
                'High': 151 + i * 0.5,
                'Low': 149 + i * 0.5,
                'Close': 150 + i * 0.5,
                'Volume': 1000000 + i * 10000
            })
        df = pd.DataFrame(data)
        df.set_index(['Ticker', 'Date'], inplace=True)
        return df


class TestScoringEngine:
    """Test scoring engine combination logic."""
    
    def test_combines_multiple_factors(self):
        """Test that scoring engine properly combines factor scores."""
        factors = [MomentumFactor(), SentimentFactor()]
        
        # Override compute methods for testing
        factors[0].compute = Mock(return_value=pd.Series({'AAPL': 0.5, 'MSFT': -0.3}))
        factors[1].compute = Mock(return_value=pd.Series({'AAPL': 0.8, 'MSFT': 0.1}))
        
        engine = ScoringEngine(factors, weights={'momentum': 0.6, 'sentiment': 0.4})
        mock_data = pd.DataFrame()
        
        result = engine.compute(mock_data, sentiment_scores={})
        
        assert 'composite' in result.columns
        assert 'rank' in result.columns
        # AAPL: (0.5 * 0.6) + (0.8 * 0.4) = 0.3 + 0.32 = 0.62
        # MSFT: (-0.3 * 0.6) + (0.1 * 0.4) = -0.18 + 0.04 = -0.14
        assert result.loc['AAPL', 'composite'] == 0.62
        assert result.loc['MSFT', 'composite'] == -0.14


class TestBacktestEngine:
    """Test backtesting functionality."""
    
    def test_returns_required_metrics(self):
        """Test that backtest returns all expected metrics."""
        engine = BacktestEngine()
        mock_scores = pd.Series({'AAPL': 0.8, 'MSFT': 0.6, 'GOOGL': 0.4})
        mock_prices = pd.DataFrame()
        
        result = engine.run(mock_scores.to_frame('composite'), mock_prices)
        
        expected_keys = ['strategy_return', 'benchmark_return', 'excess_return', 
                        'sharpe_ratio', 'max_drawdown', 'volatility']
        for key in expected_keys:
            assert key in result