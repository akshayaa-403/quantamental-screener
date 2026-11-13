"""Tests for sentiment scoring and news retrieval (network-free)."""
from unittest.mock import Mock

import pandas as pd
import pytest

from data_sources import yahoo_finance
from data_sources.yahoo_finance import YahooFinanceSource, _news_query
from sentiment.ensemble_sentiment import EnsembleSentiment
from sentiment.vader_model import VADERModel


class TestVaderFinanceLexicon:
    @pytest.mark.parametrize("headline, sign", [
        ("Apple beats estimates", 1),
        ("Analyst upgrades Microsoft", 1),
        ("Morgan Stanley downgrades Tesla", -1),
        ("Intel misses revenue forecast", -1),
    ])
    def test_market_language_has_the_right_sign(self, headline, sign):
        assert VADERModel().analyze([headline])["score"] * sign > 0


class TestShrinkage:
    def test_few_headlines_are_shrunk_toward_neutral(self):
        model = EnsembleSentiment(prior_headlines=3)
        one = model.analyze(["Apple beats estimates"])
        assert one["n_headlines"] == 1
        assert one["score"] == pytest.approx(one["raw_score"] * 1 / 4)

    def test_more_headlines_shrink_less(self):
        model = EnsembleSentiment(prior_headlines=3)
        many = model.analyze(["Apple beats estimates"] * 9)
        assert many["score"] == pytest.approx(many["raw_score"] * 9 / 12)


class TestNewsQuery:
    @pytest.mark.parametrize("name, query", [
        ("Apple Inc.", '"Apple"'),
        ("Alphabet Inc. (Class A)", '"Alphabet"'),
        ("The Coca-Cola Company", '"Coca-Cola"'),
        ("Goldman Sachs Group, Inc.", '"Goldman Sachs"'),
        ("Johnson & Johnson", '"Johnson & Johnson"'),
    ])
    def test_company_name_without_legal_suffixes(self, name, query):
        assert _news_query("XYZ", name) == query

    def test_short_ticker_without_name_skips_newsapi(self):
        assert _news_query("T", None) is None
        assert _news_query("AAPL", None) == "AAPL"


class TestHeadlines:
    def _source(self, monkeypatch, news):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        monkeypatch.setattr(yahoo_finance.yf, "Ticker", lambda t: Mock(news=news))
        return YahooFinanceSource()

    def test_yahoo_news_is_filtered_by_publish_date(self, monkeypatch):
        now = pd.Timestamp.now(tz="UTC")
        news = [
            {"content": {"title": "Fresh", "pubDate": (now - pd.Timedelta(days=1)).isoformat()}},
            {"content": {"title": "Stale", "pubDate": (now - pd.Timedelta(days=30)).isoformat()}},
            {"title": "Old format, fresh", "providerPublishTime": int((now - pd.Timedelta(hours=5)).timestamp())},
        ]
        source = self._source(monkeypatch, news)
        assert source.get_news_headlines("AAPL", lookback_days=7) == ["Fresh", "Old format, fresh"]

    def test_newsapi_searches_titles_for_the_company_in_the_window(self, monkeypatch):
        calls = {}

        class FakeClient:
            def __init__(self, api_key):
                pass

            def get_everything(self, **kwargs):
                calls.update(kwargs)
                return {"articles": [{"title": "Apple beats"}, {"title": "apple beats"}, {"title": "[Removed]"}]}

        monkeypatch.setenv("NEWS_API_KEY", "key")
        monkeypatch.setattr(yahoo_finance, "NewsApiClient", FakeClient)
        headlines = YahooFinanceSource().get_news_headlines("AAPL", lookback_days=7, company_name="Apple Inc.")
        assert headlines == ["Apple beats"]
        assert calls["qintitle"] == '"Apple"'
        since = pd.Timestamp(calls["from_param"], tz="UTC")
        assert pd.Timedelta(days=6.9) < pd.Timestamp.now(tz="UTC") - since < pd.Timedelta(days=7.1)
