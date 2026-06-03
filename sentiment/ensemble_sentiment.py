from core.sentiment_model import SentimentModel
from .finbert_model import FinBERTModel
from .vader_model import VADERModel
from .textblob_model import TextBlobModel
from typing import List, Dict
import numpy as np

class EnsembleSentiment(SentimentModel):
    def __init__(self, models=None, weights=None):
        self.models = models or [FinBERTModel(), VADERModel(), TextBlobModel()]
        self.weights = weights or [0.5, 0.3, 0.2]  # FinBERT highest confidence
    
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        scores = []
        confidences = []
        for model in self.models:
            result = model.analyze(texts)
            scores.append(result['score'])
            confidences.append(result['confidence'])
        weighted_score = np.average(scores, weights=self.weights)
        avg_confidence = np.average(confidences, weights=self.weights)
        return {'score': weighted_score, 'confidence': avg_confidence, 'model': 'ensemble'}