"""Flask web app for sentiment analysis on localhost:5000."""

from flask import Flask, render_template, request

from sentiment_analysis import sentiment_analyzer

app = Flask("Sentiment Analyzer")


def _format_label(label):
    """Turn positive into Positive for the web UI."""
    return label.title() if label else label


@app.get("/")
def index():
    """Show the feedback form."""
    return render_template("index.html")


@app.post("/")
def analyze():
    """Run sentiment analysis on submitted feedback."""
    text = request.form.get("textToAnalyze", "").strip()
    if not text:
        return render_template(
            "index.html",
            error="Please enter some text to analyze.",
        )

    response = sentiment_analyzer(text)
    return render_template(
        "index.html",
        text=text,
        result={
            "sentiment": _format_label(response["label"]),
            "confidence": round(response["score"] * 100),
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
