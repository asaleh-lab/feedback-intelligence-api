# Feedback Intelligence API

A small Flask app that labels customer feedback as positive, negative, or neutral. It runs DistilBERT on your machine, so you do not need an external NLP API or API keys.

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

Open [http://localhost:5000](http://localhost:5000), paste some text, and submit it.

You can also use it from Python:

```python
from sentiment_analysis import sentiment_analyzer

sentiment_analyzer("I love this new technology.")
```

The HTTP route is `GET /sentimentAnalyzer?textToAnalyze=...`.

## How it works

The app uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model. That model is binary, so when it is unsure we treat the result as neutral. `server.py` handles the request and returns a plain sentence with the label and score.
