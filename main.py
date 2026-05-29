#!/usr/bin/env python3
"""
Quantamental Equity Screener – Main entry point.
Runs the complete screening pipeline: universe selection, data download,
technical analysis, sentiment analysis, scoring, visualisation, backtesting,
and report generation.
"""
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config, config
from quantamental.universe import UniverseSelector
from quantamental.data_collector import DataCollector
from quantamental.sentiment import SentimentAnalyzer
from quantamental.scoring import ScoringEngine
from quantamental.visualizer import Visualizer
from quantamental.backtest import BacktestEngine
from quantamental.utils import save_results, generate_report, analyze_individual_stock


def main():
    print("=" * 80)
    print("QUANTAMENTAL EQUITY SCREENER")
    print("=" * 80)

    # 1. Universe selection
    print("\n1. Selecting stock universe...")
    universe_selector = UniverseSelector(config)
    universe = universe_selector.get_universe()
    print(f"   Universe size: {len(universe)} stocks")
    print(f"   Sample: {universe[:5]}")

    # 2. Data collection and technical analysis
    print("\n2. Downloading price data and computing technical indicators...")
    data_collector = DataCollector(config)
    price_data = data_collector.download_price_data(universe)
    technical_data = data_collector.process_all_technical_data()

    # 3. Sentiment analysis
    print("\n3. Performing news sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer(config)
    # Limit to stocks with price data (or use full universe)
    stocks_for_sentiment = list(price_data.keys())[:20]  # First 20 for speed
    sentiment_data = sentiment_analyzer.analyze_all_stocks(stocks_for_sentiment)

    # 4. Scoring
    print("\n4. Scoring stocks...")
    scoring_engine = ScoringEngine(config, data_collector, sentiment_data, universe_selector)
    available_tickers = list(set(price_data.keys()) & set(sentiment_data.keys()))
    scores_df = scoring_engine.score_all_stocks(available_tickers)
    top_stocks = scoring_engine.get_top_stocks(scores_df)

    # 5. Visualisation
    print("\n5. Generating visualisation dashboard...")
    visualizer = Visualizer(config, price_data)
    visualizer.create_dashboard(scores_df, top_stocks)

    # 6. Backtesting
    print("\n6. Running backtest...")
    backtest_engine = BacktestEngine(config)
    # Use a subset of universe for backtest (first 10)
    backtest_universe = list(price_data.keys())[:10]
    historical_data = backtest_engine.get_historical_universe(6, backtest_universe)
    backtest_results = backtest_engine.run_backtest(historical_data)
    backtest_viz = backtest_engine.create_backtest_visualization(backtest_results)
    backtest_viz.show()

    # 7. Report and exports
    print("\n7. Generating final report...")
    generate_report(scores_df, top_stocks, backtest_results, sentiment_data)
    save_results(top_stocks, scores_df, config)

    # 8. Individual stock analysis for top 3
    print("\n8. Individual stock deep dive...")
    for ticker in top_stocks['ticker'].head(3):
        analyze_individual_stock(ticker, scores_df, data_collector, sentiment_data, universe_selector)

    print("\n" + "=" * 80)
    print("SCREENING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()