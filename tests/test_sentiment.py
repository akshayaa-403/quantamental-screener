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

    def test_yahoo_articles_carry_link_and_outlet_tag(self, monkeypatch):
        now = pd.Timestamp.now(tz="UTC")
        news = [
            {"content": {"title": "New", "pubDate": now.isoformat(), "provider": {"displayName": "Reuters"},
                         "canonicalUrl": {"url": "https://www.reuters.com/a"}}},
            {"title": "Old", "providerPublishTime": int(now.timestamp()), "publisher": "The Wall Street Journal",
             "link": "https://www.wsj.com/b"},
        ]
        articles = self._source(monkeypatch, news).get_news_articles("AAPL", lookback_days=7)
        assert [(a["url"], a["tag"]) for a in articles] == [("https://www.reuters.com/a", "Reuters"),
                                                              ("https://www.wsj.com/b", "WSJ")]

    def test_empty_news_is_cached_briefly(self, monkeypatch):
        source = self._source(monkeypatch, [])
        source._cache = Mock(get=Mock(return_value=None))
        assert source.get_news_articles("AAPL") == []
        assert source._cache.set.call_args.kwargs["ttl"] == 300

    def test_get_news_for_skips_names_without_news_or_with_errors(self, monkeypatch):
        source = YahooFinanceSource()
        news = {"AAPL": ["Apple beats"], "MSFT": []}

        def headlines(ticker, *args):
            if ticker == "BAD":
                raise RuntimeError("boom")
            return news[ticker]
        monkeypatch.setattr(source, "get_news_headlines", headlines)
        assert source.get_news_for(["AAPL", "MSFT", "BAD"], {}, 7, 10) == {"AAPL": ["Apple beats"]}

    def test_source_tag(self):
        assert yahoo_finance.source_tag("The New York Times") == "NYT"
        assert yahoo_finance.source_tag(None, "https://www.nytimes.com/x") == "NYT"
        assert yahoo_finance.source_tag("Some Blog") == "Some Blog"
        assert yahoo_finance.source_tag(None, "https://www.example.com/x") == "example.com"
        assert yahoo_finance.source_tag(None, None) is None

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
