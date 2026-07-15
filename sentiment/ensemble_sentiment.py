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

    def __init__(self, models=None, weights=None):
        self.models = None
        self.model_classes = models
        self.weights = weights or [0.6, 0.4]
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
        weighted_score = np.average(scores, weights=self.weights)
        avg_confidence = np.average(confidences, weights=self.weights)
        return {'score': weighted_score, 'confidence': avg_confidence, 'model': 'ensemble'}