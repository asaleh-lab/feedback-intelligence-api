"""Unit tests for the sentiment analyzer."""

import unittest

from sentiment_analysis import sentiment_analyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Tests for sentiment_analyzer output labels."""

    def test_sentiment_analyzer(self):
        """Positive, negative, and neutral text return expected labels."""
        self.assertEqual(
            sentiment_analyzer("I love working with Python")["label"],
            "positive",
        )
        self.assertEqual(
            sentiment_analyzer("I hate working with Python")["label"],
            "negative",
        )
        self.assertEqual(
            sentiment_analyzer("Okay I guess")["label"],
            "neutral",
        )


if __name__ == "__main__":
    unittest.main()
