"""
Utility functions – data export, individual stock analysis, report generation.
"""
import json
from datetime import datetime
import pandas as pd
import numpy as np


def save_results(top_stocks: pd.DataFrame, scores_df: pd.DataFrame, config) -> None:
    """Save all results for future use"""
    print("\nSaving results...")
    top_stocks.to_csv('top_stocks_recommendations.csv', index=False)
    scores_df.to_csv('all_stock_scores.csv', index=False)

    config_dict = {
        'universe_size': config.UNIVERSE_SIZE,
        'weights': {
            'momentum': config.MOMENTUM_WEIGHT,
            'sentiment': config.SENTIMENT_WEIGHT,
            'volume': config.VOLUME_WEIGHT,
            'volatility': config.VOLATILITY_WEIGHT
        },
        'analysis_date': datetime.now().isoformat(),
        'stocks_analyzed': len(scores_df),
        'top_n': len(top_stocks)
    }
    with open('screener_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print("Results saved:")
    print("   • top_stocks_recommendations.csv")
    print("   • all_stock_scores.csv")
    print("   • screener_config.json")


def generate_report(scores_df: pd.DataFrame, top_stocks: pd.DataFrame,
                   backtest_results: dict = None, sentiment_data: dict = None) -> None:
    """Generate comprehensive report"""
    print("\n" + "="*80)
    print("QUANTAMENTAL EQUITY SCREENER - FINAL REPORT")
    print("="*80)

    # Universe statistics
    print(f"\nUNIVERSE ANALYSIS:")
    print(f"   Total stocks analyzed: {len(scores_df)}")
    print(f"   Sectors represented: {scores_df['sector'].nunique()}")
    print(f"   Average composite score: {scores_df['composite_score'].mean():.3f}")
    print(f"   Score std deviation: {scores_df['composite_score'].std():.3f}")

    # Top stocks details
    print(f"\nTOP {len(top_stocks)} STOCK RECOMMENDATIONS:")
    print(f"{'Rank':<4} {'Ticker':<8} {'Score':<8} {'Momentum':<10} {'Sentiment':<10} {'Sector':<20}")
    print("-" * 70)
    for _, row in top_stocks.iterrows():
        print(f"{row['rank']:<4} {row['ticker']:<8} {row['composite_score']:<8.3f} "
              f"{row['momentum_score']:<10.3f} {row['sentiment_score']:<10.3f} {row['sector']:<20}")

    # Factor correlations
    print(f"\nFACTOR ANALYSIS:")
    factor_cols = ['momentum_score', 'sentiment_score', 'volume_score', 'volatility_score']
    correlations = scores_df[factor_cols].corr()
    for i, f1 in enumerate(factor_cols):
        for f2 in factor_cols[i+1:]:
            print(f"   {f1} vs {f2}: {correlations.loc[f1, f2]:.3f}")

    # Sector distribution
    print(f"\nSECTOR DISTRIBUTION:")
    sector_dist = scores_df['sector'].value_counts()
    for sector, count in sector_dist.items():
        print(f"   {sector}: {count} stocks ({count/len(scores_df)*100:.1f}%)")

    # Backtest results
    if backtest_results:
        print(f"\nBACKTEST PERFORMANCE:")
        print(f"   Strategy Return: {backtest_results['total_return']:.2%}")
        print(f"   Benchmark Return: {backtest_results['benchmark_return']:.2%}")
        print(f"   Excess Return: {backtest_results['excess_return']:.2%}")
        print(f"   Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"   Volatility: {backtest_results['volatility']:.2%}")

    # Insights
    print(f"\nKEY INSIGHTS & RECOMMENDATIONS:")
    top_sector = top_stocks['sector'].value_counts().index[0]
    top_sector_count = top_stocks['sector'].value_counts().iloc[0]
    print(f"   • {top_sector} sector dominates top picks ({top_sector_count} stocks)")
    high_score_count = len(scores_df[scores_df['composite_score'] > 0.1])
    print(f"   • {high_score_count} stocks show strong positive signals ({high_score_count/len(scores_df)*100:.1f}%)")
    avg_momentum = top_stocks['momentum_score'].mean()
    avg_sentiment = top_stocks['sentiment_score'].mean()
    if avg_momentum > avg_sentiment:
        print(f"   • Technical momentum is the primary driver (avg: {avg_momentum:.3f})")
    else:
        print(f"   • News sentiment is the primary driver (avg: {avg_sentiment:.3f})")

    print(f"\nRISK WARNINGS:")
    news_count = len(sentiment_data) if sentiment_data else 0
    print(f"   • This analysis is based on {news_count} stocks with news data")
    print(f"   • Sentiment analysis may have inherent biases")
    print(f"   • Past performance does not guarantee future results")
    print(f"   • Always conduct additional due diligence before investing")
    print("\n" + "="*80)


def analyze_individual_stock(ticker: str, scores_df: pd.DataFrame, data_collector,
                            sentiment_data: dict, universe_selector) -> None:
    """Provide detailed analysis for individual stock"""
    print(f"\nDETAILED ANALYSIS: {ticker}")
    print("=" * 50)

    if ticker not in scores_df['ticker'].values:
        print(f"{ticker} not found in analyzed universe")
        return

    stock_row = scores_df[scores_df['ticker'] == ticker].iloc[0]
    print(f"\nSCORING BREAKDOWN:")
    print(f"   Rank: #{stock_row['rank']}")
    print(f"   Composite Score: {stock_row['composite_score']:.3f}")
    print(f"   Momentum Score: {stock_row['momentum_score']:.3f}")
    print(f"   Sentiment Score: {stock_row['sentiment_score']:.3f}")
    print(f"   Volume Score: {stock_row['volume_score']:.3f}")
    print(f"   Volatility Score: {stock_row['volatility_score']:.3f}")
    print(f"   Sector: {stock_row['sector']}")

    if sentiment_data and ticker in sentiment_data:
        sent_data = sentiment_data[ticker]
        print(f"\nNEWS SENTIMENT:")
        print(f"   Overall Sentiment: {sent_data['overall_sentiment']:.3f}")
        print(f"   Articles Analyzed: {sent_data['article_count']}")
        print(f"   Recent Headlines:")
        for article in sent_data['articles'][:3]:
            print(f"     • {article['headline'][:70]}... (Score: {article['sentiment_score']:.2f})")

    latest_values = data_collector.get_latest_values(ticker)
    if latest_values:
        print(f"\nTECHNICAL INDICATORS:")
        price = latest_values.get('price', 0)
        if isinstance(price, pd.Series):
            price = price.iloc[-1]
        rsi = latest_values.get('rsi', 0)
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[-1]
        macd = latest_values.get('macd', 0)
        if isinstance(macd, pd.Series):
            macd = macd.iloc[-1]
        volume = latest_values.get('volume', 0)
        if isinstance(volume, pd.Series):
            volume = volume.iloc[-1]
        volume_sma = latest_values.get('volume_sma', 1)
        if isinstance(volume_sma, pd.Series):
            volume_sma = volume_sma.iloc[-1]
        print(f"   Price: ${float(price):.2f}")
        print(f"   RSI: {float(rsi):.1f}")
        print(f"   MACD: {float(macd):.3f}")
        if volume_sma != 0:
            print(f"   Volume Ratio: {float(volume)/float(volume_sma):.2f}x")
        else:
            print("   Volume Ratio: N/A")