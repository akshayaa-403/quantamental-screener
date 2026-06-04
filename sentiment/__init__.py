# Lazy import to avoid loading heavy models at module import time
def __getattr__(name):
    if name == "FinBERTModel":
        from .finbert_model import FinBERTModel
        return FinBERTModel
    elif name == "VADERModel":
        from .vader_model import VADERModel
        return VADERModel
    elif name == "TextBlobModel":
        from .textblob_model import TextBlobModel
        return TextBlobModel
    elif name == "EnsembleSentiment":
        from .ensemble_sentiment import EnsembleSentiment
        return EnsembleSentiment
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["FinBERTModel", "VADERModel", "TextBlobModel", "EnsembleSentiment"]