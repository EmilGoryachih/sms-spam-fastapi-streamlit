# SMS Spam Classifier — FastAPI + Streamlit

A compact end-to-end demo for deploying a text classifier:
- **Model**: TF-IDF + Logistic Regression (ham/spam)
- **API**: FastAPI (`/health`, `/predict`)
- **UI**: Streamlit (minimal, modern dark theme)
- **Containers**: separate images for API and UI, orchestrated via Docker Compose

---

## Repository structure

```

.
├─ code/
│  ├─ models/
│  │  └─ train.py              # train and serialize model -> models/model.pkl
│  └─ deployment/
│     ├─ api/
│     │  ├─ app.py             # FastAPI app
│     │  └─ Dockerfile         # API image
│     ├─ app/
│     │  ├─ streamlit_app.py   # Streamlit UI
│     │  └─ Dockerfile         # UI image
│     └─ docker-compose.yml    # compose file (lives here; build context is repo root)
├─ data/
│  └─ raw/
│     └─ spam.csv              # raw dataset (UCI/Kaggle SMS Spam)
├─ models/
│  └─ model.pkl                # trained model will be saved here
├─ requirements.txt
└─ README.md

````

---

## Prerequisites

- Docker Desktop (with Docker Compose)
- Python 3.10+ (only needed if you want to train/run locally without Docker)

---

## Quick start (Docker Compose)

> The compose file is located in `code/deployment` and builds with **context = repo root**.

```bash
# from the repo root
cd code/deployment

# build and start both services (API + UI)
docker compose up --build
````

Open:

* API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* UI:       [http://localhost:8501](http://localhost:8501)

Stop:

```bash
docker compose down
```

---

## Train the model (optional if `models/model.pkl` already exists)

```bash
# from the repo root
python -m pip install -r requirements.txt

# train on data/raw/spam.csv and save model to models/model.pkl
python code/models/train.py \
  --raw data/raw/spam.csv \
  --out models/model.pkl \
  --metrics-out models/metrics.json
```

Artifacts:

* `models/model.pkl` — serialized sklearn Pipeline (TF-IDF + LogisticRegression)
* `models/metrics.json` — hold-out metrics (accuracy, precision, recall, f1, roc\_auc)

---

## Run services without Docker (local dev)

**API**

```bash
python -m pip install -r requirements.txt
uvicorn code.deployment.api.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs
```

**UI**

```bash
# by default UI calls http://fastapi:8000/predict in Docker.
# locally, point it to your localhost API:
export FASTAPI_URL=http://localhost:8000/predict  # PowerShell: $env:FASTAPI_URL="http://localhost:8000/predict"
streamlit run code/deployment/app/streamlit_app.py
# open http://localhost:8501
```

---

## API contract

**Health**

```http
GET /health  ->  200 OK
{"status": "ok"}
```

**Predict**

* Request JSON: `{"text": "<sms message>"}`
* Response JSON: `{"label": "spam" | "ham", "proba": <float in [0,1]>}`

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"WINNER!! You have won a FREE prize. Call now!"}'
```

---

## Configuration

UI environment variables:

* `FASTAPI_URL` — API endpoint for predictions. Default inside Docker: `http://fastapi:8000/predict`.
  For local dev set `http://localhost:8000/predict`.
* `REQUEST_TIMEOUT` — HTTP timeout in seconds (default: `6.0`).

API environment variables:

* `MODEL_PATH` — path to the model file (default: `models/model.pkl` inside the image).

Ports:

* API: `8000`
* UI:  `8501`