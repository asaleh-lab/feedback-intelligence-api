"""Core sentiment analysis logic using a local Hugging Face model."""

import functools

from transformers import pipeline

MARGIN_THRESHOLD = 0.35
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@functools.lru_cache(maxsize=1)
def _get_classifier():
    """Load the sentiment pipeline once and reuse it."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        top_k=None,
    )


def _scores_by_label(scores):
    """Return label probabilities keyed by normalized label name."""
    return {item["label"].lower(): item["score"] for item in scores}


def _map_label(by_label):
    """Map model probabilities to the app's three-label format."""
    positive_score = by_label.get("positive", 0.0)
    negative_score = by_label.get("negative", 0.0)
    neutral_score = by_label.get("neutral", 0.0)

    if neutral_score > 0:
        best_label = "SENT_NEUTRAL"
        best_score = neutral_score
        if positive_score > best_score:
            best_label = "SENT_POSITIVE"
            best_score = positive_score
        if negative_score > best_score:
            best_label = "SENT_NEGATIVE"
            best_score = negative_score
        return {"label": best_label, "score": best_score}

    margin = abs(positive_score - negative_score)
    if margin < MARGIN_THRESHOLD:
        return {
            "label": "SENT_NEUTRAL",
            "score": max(positive_score, negative_score),
        }
    if positive_score > negative_score:
        return {"label": "SENT_POSITIVE", "score": positive_score}
    return {"label": "SENT_NEGATIVE", "score": negative_score}


def sentiment_analyzer(text_to_analyse):
    """
    Run sentiment analysis on the given text.

    Returns a dict with ``label`` and ``score``.
    """
    scores = _get_classifier()(text_to_analyse)[0]
    return _map_label(_scores_by_label(scores))
