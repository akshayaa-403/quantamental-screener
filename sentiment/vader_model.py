from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.sentiment_model import SentimentModel
from typing import List, Dict
import numpy as np

# VADER's lexicon comes from social media and misses most market language:
# "beats estimates", "downgrade" or "plunges" score as neutral. These entries
# (VADER's -4..+4 scale) cover the common headline verbs and nouns, in the
# spirit of the Loughran-McDonald finance word lists.
FINANCE_LEXICON = {
    # positive
    "beat": 1.8, "beats": 1.8, "topped": 1.5, "tops": 1.5, "exceeds": 1.6, "exceeded": 1.6,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0, "outperform": 1.8, "outperforms": 1.8,
    "overweight": 1.2, "bullish": 2.0, "rally": 1.6, "rallies": 1.6, "rallied": 1.6,
    "surge": 1.8, "surges": 1.8, "surged": 1.8, "soar": 2.0, "soars": 2.0, "soared": 2.0,
    "jump": 1.3, "jumps": 1.3, "jumped": 1.3, "climb": 1.0, "climbs": 1.0, "gain": 1.2, "gains": 1.2,
    "record": 1.0, "buyback": 1.2, "raises": 1.0, "raised": 1.0, "dividend": 0.6, "profit": 1.0,
    "profitable": 1.4, "growth": 1.0, "expands": 0.8, "approval": 1.5, "approved": 1.5,
    # negative
    "miss": -1.8, "misses": -1.8, "missed": -1.8, "downgrade": -2.0, "downgrades": -2.0,
    "downgraded": -2.0, "underperform": -1.8, "underweight": -1.2, "bearish": -2.0,
    "plunge": -2.2, "plunges": -2.2, "plunged": -2.2, "tumble": -1.8, "tumbles": -1.8, "tumbled": -1.8,
    "slump": -1.8, "slumps": -1.8, "sink": -1.5, "sinks": -1.5, "slide": -1.2, "slides": -1.2,
    "fall": -1.2, "falls": -1.2, "fell": -1.2, "drop": -1.2, "drops": -1.2, "dropped": -1.2,
    "decline": -1.2, "declines": -1.2, "cut": -1.0, "cuts": -1.0, "layoffs": -1.8, "lawsuit": -1.6,
    "probe": -1.5, "investigation": -1.5, "recall": -1.5, "loss": -1.5, "losses": -1.5,
    "warns": -1.6, "warning": -1.4, "bankruptcy": -3.0, "default": -2.2, "fraud": -3.0,
    "selloff": -1.8, "sell-off": -1.8, "volatile": -0.6, "weak": -1.2, "weaker": -1.2,
    "lowers": -1.0, "lowered": -1.0, "delay": -1.0, "delayed": -1.0, "fine": -0.8, "fined": -1.6,
}


class VADERModel(SentimentModel):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(FINANCE_LEXICON)

    def analyze(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {'score': 0.0, 'confidence': 0.5, 'model': 'vader'}
        scores = [self.analyzer.polarity_scores(text)['compound'] for text in texts]
        avg_score = np.mean(scores)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7
        return {'score': avg_score, 'confidence': confidence, 'model': 'vader'}
