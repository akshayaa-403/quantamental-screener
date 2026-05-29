"""
Data collection and technical analysis – downloads price data and computes all technical indicators.
"""
import time
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from config.settings import Config


class DataCollector:
    """Collects and processes financial data"""

    def __init__(self, config: Config):
        self.config = config
        self.price_data = {}
        self.technical_data = {}

    def download_price_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Download price data for all tickers"""
        print(f"Downloading price data for {len(tickers)} stocks...")

        for i, ticker in enumerate(tickers):
            if i % 10 == 0:
                print(f"   Downloaded {i}/{len(tickers)}")

            try:
                data = yf.download(ticker, period=self.config.DATA_PERIOD, progress=False)
                if len(data) > 50:  # Ensure minimum data points
                    self.price_data[ticker] = data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue

        print(f"Successfully downloaded data for {len(self.price_data)} stocks")
        return self.price_data

    def calculate_technical_indicators(self, ticker: str, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for a single stock"""
        # extract series and force 1-D
        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()
        volume = data['Volume'].squeeze()

        try:
            indicators = {}

            # Price-based indicators
            indicators['sma_20'] = ta.trend.sma_indicator(close, window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(close, window=50)
            indicators['ema_12'] = ta.trend.ema_indicator(close, window=12)
            indicators['ema_26'] = ta.trend.ema_indicator(close, window=26)

            # Momentum indicators
            indicators['rsi'] = ta.momentum.rsi(close, window=self.config.RSI_PERIOD)
            indicators['stoch_rsi'] = ta.momentum.stochrsi(close)

            # MACD
            macd = ta.trend.MACD(close,
                                 window_fast=self.config.MACD_FAST,
                                 window_slow=self.config.MACD_SLOW,
                                 window_sign=self.config.MACD_SIGNAL)
            indicators['macd'] = macd.macd()
            indicators['macd_signal'] = macd.macd_signal()
            indicators['macd_histogram'] = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close,
                                              window=self.config.BB_PERIOD,
                                              window_dev=self.config.BB_STD)
            indicators['bb_upper'] = bb.bollinger_hband()
            indicators['bb_middle'] = bb.bollinger_mavg()
            indicators['bb_lower'] = bb.bollinger_lband()
            indicators['bb_width'] = bb.bollinger_wband()

            # Volume indicators
            indicators['volume_sma'] = ta.trend.sma_indicator(volume, window=20)
            indicators['cmf'] = ta.volume.chaikin_money_flow(high, low, close, volume)

            # Volatility indicators
            indicators['atr'] = ta.volatility.average_true_range(high, low, close)

            # Trend indicators
            indicators['adx'] = ta.trend.adx(high, low, close)

            return indicators

        except Exception as e:
            print(f"Error calculating indicators for {ticker}: {e}")
            return {}

    def process_all_technical_data(self) -> Dict[str, Dict]:
        """Process technical indicators for all stocks"""
        print("Calculating technical indicators...")

        for i, (ticker, data) in enumerate(self.price_data.items()):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(self.price_data)}")

            indicators = self.calculate_technical_indicators(ticker, data)
            if indicators:
                self.technical_data[ticker] = indicators

        print(f"Technical indicators calculated for {len(self.technical_data)} stocks")
        return self.technical_data

    def get_latest_values(self, ticker: str) -> Dict:
        """Get latest values for all indicators"""
        if ticker not in self.technical_data:
            return {}

        latest_values = {}
        for indicator, series in self.technical_data[ticker].items():
            if len(series) > 0:
                latest_values[indicator] = series.iloc[-1]

        # Add price data
        if ticker in self.price_data:
            price_data = self.price_data[ticker]
            latest_values['price'] = price_data['Close'].iloc[-1]
            latest_values['volume'] = price_data['Volume'].iloc[-1]
            latest_values['high_52w'] = price_data['High'].rolling(252).max().iloc[-1]
            latest_values['low_52w'] = price_data['Low'].rolling(252).min().iloc[-1]

        return latest_values