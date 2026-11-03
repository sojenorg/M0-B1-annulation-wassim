"""API de prédiction d'annulation — Escale Atlantique (mission FastIA)."""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.schemas import Booking, ModelInfo, PredictionResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "model" / "cancellation_model.joblib"))
INFO_PATH = MODEL_PATH.with_name("model_info.json")
LOG_PATH = ROOT_DIR / "logs" / "api.log"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"

# remove() d'abord pour ne pas dupliquer les sinks si le module est rechargé
logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
logger.add(
    LOG_PATH,
    level="INFO",
    format=LOG_FORMAT,
    rotation="5 MB",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
)

# Objets partagés par toutes les requêtes (ici : le modèle)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exécuté UNE fois au démarrage : on charge le modèle en mémoire
    state["model"] = joblib.load(MODEL_PATH)
    state["info"] = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    logger.info("Modèle chargé depuis {}", MODEL_PATH)
    yield
    # Exécuté à l'arrêt du serveur
    state.clear()


app = FastAPI(
    title="Escale Atlantique — Prédiction d'annulation",
    description="Estime la probabilité qu'une réservation soit annulée.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Le service est-il en vie, et le modèle chargé ?"""
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/info", response_model=ModelInfo)
def info() -> ModelInfo:
    """Identité du modèle chargé."""
    return ModelInfo(**state["info"])


def run_model(bookings: list[Booking]) -> list[PredictionResponse]:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    if not bookings:
        return []

    features = pd.DataFrame([b.model_dump() for b in bookings])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        PredictionResponse(annulation=bool(p), probabilite_annulation=round(float(q), 4))
        for p, q in zip(predictions, probabilities)
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(booking: Booking) -> PredictionResponse:
    """Prédit l'annulation d'une réservation."""
    start = time.perf_counter()
    response = run_model([booking])[0]
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict | entrée={} | annulation={} | probabilité={} | durée={:.1f} ms",
        booking.model_dump(),
        response.annulation,
        response.probabilite_annulation,
        duration_ms,
    )
    return response


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(bookings: list[Booking]) -> list[PredictionResponse]:
    """Prédit l'annulation d'une liste de réservations."""
    start = time.perf_counter()
    responses = run_model(bookings)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict/batch | n={} | annulations={} | durée={:.1f} ms",
        len(bookings),
        sum(r.annulation for r in responses),
        duration_ms,
    )
    return responses
