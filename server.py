"""Flask web app for sentiment analysis on localhost:5000."""

from flask import Flask, render_template, request

from sentiment_analysis import sentiment_analyzer

app = Flask("Sentiment Analyzer")


def _format_label(label):
    """Turn SENT_POSITIVE into POSITIVE for the web UI."""
    if label and label.startswith("SENT_"):
        return label.replace("SENT_", "", 1)
    return label


@app.route("/sentimentAnalyzer")
def sent_analyzer():
    """Analyze text from the query string and return label plus score."""
    text_to_analyze = request.args.get("textToAnalyze")

    if not text_to_analyze or not text_to_analyze.strip():
        return "Invalid input ! Try again."

    response = sentiment_analyzer(text_to_analyze)
    label = response["label"]
    score = response["score"]

    if label is None:
        return "Invalid input ! Try again."

    display_label = _format_label(label)
    return (
        f"The given text has been identified as {display_label} "
        f"with a score of {score}."
    )


@app.route("/")
def render_index_page():
    """Render the main application page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
