from core.sentiment_model import SentimentModel
from typing import List, Dict
import numpy as np

class EnsembleSentiment(SentimentModel):
    """Weighted ensemble over one or more sentiment models.

    Defaults to VADER + TextBlob: both are lightweight, rule/lexicon-based
    models with no ML weights to download or hold in memory, which keeps this
    class safe to use in memory-constrained deployments (e.g. Streamlit
    Community Cloud's ~1GB limit). FinBERT (transformers + torch) is far more
    accurate on financial text but adds several hundred MB+ of resident memory
    once loaded, so it is opt-in: pass
    ``models=[FinBERTModel, VADERModel, TextBlobModel]`` explicitly (e.g. for
    local/CLI use where memory isn't capped) to include it.
    """

    def __init__(self, models=None, weights=None, prior_headlines=None):
        self.models = None
        self.model_classes = models
        # VADER (with the finance lexicon) reads headlines far better than
        # TextBlob's general-purpose polarity, so it carries most of the weight.
        self.weights = weights or [0.7, 0.3]
        if prior_headlines is None:
            from config.settings import get_settings
            prior_headlines = get_settings().sentiment.prior_headlines
        self.prior_headlines = prior_headlines
        self._loaded = False

    def _load_models(self):
        if not self._loaded:
            if self.model_classes is None:
                from .vader_model import VADERModel
                from .textblob_model import TextBlobModel
                self.model_classes = [VADERModel, TextBlobModel]
            if len(self.model_classes) != len(self.weights):
                raise ValueError(
                    f"{len(self.model_classes)} models but {len(self.weights)} weights; "
                    "pass a matching `weights` list."
                )
            self.models = [model_class() for model_class in self.model_classes]
            self._loaded = True
    
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        self._load_models()
        scores = []
        confidences = []
        for model in self.models:
            result = model.analyze(texts)
            scores.append(result['score'])
            confidences.append(result['confidence'])
        raw_score = float(np.average(scores, weights=self.weights))
        avg_confidence = float(np.average(confidences, weights=self.weights))
        # Shrink toward neutral by the amount of evidence: the mean of n noisy
        # headline scores is trusted in proportion n / (n + prior), so a single
        # headline can't put a stock at the top or bottom of the ranking.
        n = len(texts)
        shrink = n / (n + self.prior_headlines) if n else 0.0
        return {'score': raw_score * shrink, 'raw_score': raw_score, 'n_headlines': n,
                'confidence': avg_confidence, 'model': 'ensemble'}