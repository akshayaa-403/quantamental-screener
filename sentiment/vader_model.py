from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.sentiment_model import SentimentModel
from typing import List, Dict
import numpy as np

class VADERModel(SentimentModel):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {'score': 0.0, 'confidence': 0.5, 'model': 'vader'}
        scores = [self.analyzer.polarity_scores(text)['compound'] for text in texts]
        avg_score = np.mean(scores)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7
        return {'score': avg_score, 'confidence': confidence, 'model': 'vader'}