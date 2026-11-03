"""Interface web minimale : un formulaire qui appelle l'API via httpx."""

import html
import json
import os
from pathlib import Path
from typing import get_args

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from app.schemas import Booking

API_URL = os.getenv("API_URL", "http://localhost:8000")
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "exemple_reservation.json"
EXAMPLE = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

app = FastAPI(title="Escale Atlantique — Interface")

PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Prédiction d'annulation</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  form {{ display: grid; grid-template-columns: 1fr 1fr; gap: .75rem 1.5rem; }}
  label {{ display: flex; flex-direction: column; font-size: .9rem; }}
  input, select {{ padding: .4rem; font-size: 1rem; }}
  button {{ grid-column: 1 / -1; padding: .6rem; font-size: 1rem; cursor: pointer; }}
  .result {{ margin-top: 1.5rem; padding: 1rem; border-radius: 6px; }}
  .ok {{ background: #e6f4ea; }}
  .warn {{ background: #fdecea; }}
  .err {{ background: #fff4e5; }}
  small {{ color: #666; }}
</style>
</head>
<body>
<h1>Escale Atlantique — Prédiction d'annulation</h1>
<small>API : {api_url}</small>
<form method="get" action="/predict">
{fields}
<button type="submit">Prédire</button>
</form>
{result}
</body>
</html>
"""


def field_html(name: str, value) -> str:
    annotation = Booking.model_fields[name].annotation
    options = get_args(annotation)
    if options:
        choices = "".join(
            f'<option{" selected" if o == value else ""}>{html.escape(o)}</option>' for o in options
        )
        return f'<label>{name}<select name="{name}">{choices}</select></label>'
    if annotation is str:
        return f'<label>{name}<input name="{name}" value="{html.escape(str(value))}" required></label>'
    step = "any" if annotation is float else "1"
    return f'<label>{name}<input type="number" name="{name}" value="{value}" step="{step}" required></label>'


def to_booking(params) -> dict:
    """Convertit les paramètres du formulaire (chaînes) vers les types attendus."""
    booking = {}
    for name, field in Booking.model_fields.items():
        raw = params.get(name, "")
        try:
            booking[name] = field.annotation(raw) if field.annotation in (int, float) else raw
        except ValueError:
            booking[name] = raw
    return booking


def render(values: dict, result: str = "") -> HTMLResponse:
    fields = "\n".join(field_html(name, values.get(name, "")) for name in Booking.model_fields)
    return HTMLResponse(PAGE.format(api_url=html.escape(API_URL), fields=fields, result=result))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return render(EXAMPLE)


@app.get("/predict", response_class=HTMLResponse)
def predict(request: Request) -> HTMLResponse:
    booking = to_booking(request.query_params)
    try:
        response = httpx.post(f"{API_URL}/predict", json=booking, timeout=10)
    except httpx.HTTPError as exc:
        return render(booking, f'<div class="result err">API injoignable : {html.escape(str(exc))}</div>')

    if response.status_code == 200:
        body = response.json()
        verdict = "OUI" if body["annulation"] else "NON"
        proba = body["probabilite_annulation"] * 100
        css = "warn" if body["annulation"] else "ok"
        result = f'<div class="result {css}"><strong>Annulation probable : {verdict}</strong><br>Probabilité : {proba:.1f} %</div>'
    else:
        errors = "".join(
            f"<li>{html.escape('.'.join(map(str, e['loc'][1:])))} : {html.escape(e['msg'])}</li>"
            for e in response.json().get("detail", [])
        )
        result = f'<div class="result err"><strong>Entrée refusée ({response.status_code})</strong><ul>{errors}</ul></div>'
    return render(booking, result)
