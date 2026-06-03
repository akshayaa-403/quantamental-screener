from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from core.sentiment_model import SentimentModel
from typing import List, Dict

class FinBERTModel(SentimentModel):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
    
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {'score': 0.0, 'confidence': 0.5, 'model': 'finbert'}
        # Simple average of probabilities
        scores = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # FinBERT classes: 0=negative, 1=neutral, 2=positive
            pos_prob = probs[0][2].item()
            neg_prob = probs[0][0].item()
            score = pos_prob - neg_prob  # range -1..1
            scores.append(score)
        avg_score = np.mean(scores)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.8
        return {'score': avg_score, 'confidence': confidence, 'model': 'finbert'}