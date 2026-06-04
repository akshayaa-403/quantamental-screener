from core.sentiment_model import SentimentModel
from typing import List, Dict
import numpy as np

class EnsembleSentiment(SentimentModel):
    def __init__(self, models=None, weights=None):
        self.models = None  # Lazy load models
        self.model_classes = models  # If provided externally
        self.weights = weights or [0.5, 0.3, 0.2]  # FinBERT highest confidence
        self._loaded = False
    
    def _load_models(self):
        if not self._loaded:
            if self.model_classes is None:
                # Import here to delay loading heavy dependencies
                from .finbert_model import FinBERTModel
                from .vader_model import VADERModel
                from .textblob_model import TextBlobModel
                self.model_classes = [FinBERTModel, VADERModel, TextBlobModel]
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