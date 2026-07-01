"""Core sentiment analysis logic using a local DistilBERT model."""

import functools

from transformers import pipeline

MARGIN_THRESHOLD = 0.35
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


@functools.lru_cache(maxsize=1)
def _get_classifier():
    """Load the DistilBERT pipeline once and reuse it."""
    return pipeline(
        "sentiment-analysis",
        model=MODEL_ID,
        top_k=None,
    )


def _scores_by_label(scores):
    """Return positive and negative probabilities from model output."""
    by_label = {item["label"]: item["score"] for item in scores}
    return by_label["POSITIVE"], by_label["NEGATIVE"]


def _map_label(positive_score, negative_score):
    """Map DistilBERT probabilities to the app's three-label format."""
    margin = abs(positive_score - negative_score)
    if margin < MARGIN_THRESHOLD:
        return "SENT_NEUTRAL", max(positive_score, negative_score)
    if positive_score > negative_score:
        return "SENT_POSITIVE", positive_score
    return "SENT_NEGATIVE", negative_score


def sentiment_analyzer(text_to_analyse):
    """
    Run DistilBERT sentiment analysis on the given text.

    Returns a dict with ``label`` and ``score``.
    """
    scores = _get_classifier()(text_to_analyse)[0]
    positive_score, negative_score = _scores_by_label(scores)
    label, score = _map_label(positive_score, negative_score)
    return {"label": label, "score": score}
