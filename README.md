# Feedback Intelligence API

A small Flask app that labels customer feedback as positive, negative, or neutral. It runs a Hugging Face sentiment model on your machine, so you do not need an external NLP API or API keys.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, use `python3` and `source .venv/bin/activate`.

The first run downloads the model (~250 MB). After that, everything works offline.

## Run it

```bash
python server.py
```

Open [http://localhost:5000](http://localhost:5000), paste some text, and click **Analyze**. Flask handles the form and shows the result on the same page — no JavaScript required.

You can also use it from Python:

```python
from sentiment_analysis import sentiment_analyzer

result = sentiment_analyzer("I love this new technology.")
# {"label": "positive", "score": 0.98}
print(result["label"], result["score"])
```

## How it works

The app uses Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment-latest` model through the `transformers` pipeline. The model returns probabilities for positive, negative, and neutral labels.

`sentiment_analysis/analyzer.py` maps those scores to `positive`, `negative`, or `neutral` and returns a dict with `label` and `score`. When the model reports a neutral score, the highest of the three probabilities wins. Otherwise, if positive and negative scores are too close (within a margin threshold), the result is treated as neutral.

`server.py` serves a simple HTML form. When you submit feedback, Flask runs the analyzer and renders the result below the form.



> I’ll revisit this and add a layer focused on improving decision logic using confidence thresholds, margin rules, and entropy-based uncertainty detection.