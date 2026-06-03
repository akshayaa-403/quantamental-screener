from textblob import TextBlob
from core.sentiment_model import SentimentModel
from typing import List, Dict
import numpy as np

class TextBlobModel(SentimentModel):
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {'score': 0.0, 'confidence': 0.5, 'model': 'textblob'}
        scores = [TextBlob(text).sentiment.polarity for text in texts]
        avg_score = np.mean(scores)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.6
        return {'score': avg_score, 'confidence': confidence, 'model': 'textblob'}