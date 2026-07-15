# Quantamental Screener

Multi‑factor stock screener with sentiment analysis, technical indicators, and Redis caching.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Features

- S&P 500 universe selection with market cap/volume filters
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Ensemble sentiment analysis (VADER + TextBlob by default; optional FinBERT)
- Pluggable factor scoring system
- Historical backtesting with rebalancing
- Interactive Streamlit dashboard
- **Redis caching** for faster repeated runs

## Quick Start

```bash
# Clone repository
git clone https://github.com/akshayaa-403/quantamental-screener.git
cd quantamental-screener

# Install dependencies
pip install -r requirements.txt

# Start Redis (optional but recommended)
docker-compose up -d

# Copy environment variables
cp .env.example .env

# Run the screener (CLI)
python app.py --top-n 10

# Or launch web dashboard
streamlit run streamlit_app.py
```

## Configuration

All settings are read from environment variables (see `.env.example` for the full
list). Factor weights, universe size, backtest parameters, Redis, and API keys are
configured via `FACTOR_*`, `UNIVERSE_*`, `BACKTEST_*`, `REDIS_*`, and the API-key
variables. Sentiment analysis uses real news by default; set
`USE_MOCK_SENTIMENT=true` for a deterministic offline demo.

## Development

```bash
# Install the lightweight test dependencies and run the suite
pip install -r requirements-dev.txt
pytest
```

Tests run automatically on every push and pull request via GitHub Actions
(`.github/workflows/ci.yml`).

## Sentiment models: memory footprint

By default, sentiment analysis uses **VADER + TextBlob** — lightweight,
lexicon-based models with no ML weights to download or hold in memory. This
keeps the app safe to run on memory-capped deployments (e.g. Streamlit
Community Cloud's ~1GB limit).

**FinBERT** (`sentiment/finbert_model.py`) is more accurate on financial-domain
text but pulls in `transformers` + `torch`, adding several hundred MB+ of
resident memory once loaded — enough to push a capped deployment over its
limit. It's optional and off by default:

```bash
pip install -r requirements.txt -r requirements-finbert.txt
```

```python
from sentiment import FinBERTModel, VADERModel, TextBlobModel, EnsembleSentiment
sentiment_model = EnsembleSentiment(
    models=[FinBERTModel, VADERModel, TextBlobModel],
    weights=[0.5, 0.3, 0.2],
)
```

Only use this for local/CLI runs where memory isn't constrained, not for the
deployed Streamlit app.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.