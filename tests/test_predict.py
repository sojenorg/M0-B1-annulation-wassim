import json
from pathlib import Path

import pandas as pd
import pytest

from app.main import state

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES = [
    "hotel", "lead_time", "arrival_date_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "market_segment",
    "deposit_type", "customer_type", "country", "adr", "previous_cancellations",
    "total_of_special_requests", "required_car_parking_spaces",
]


@pytest.fixture
def booking():
    return json.loads((DATA_DIR / "exemple_reservation.json").read_text())


def csv_bookings(n=5):
    df = pd.read_csv(DATA_DIR / "hotel_bookings_sample.csv")[FEATURES].dropna()
    return df.head(n).to_dict(orient="records")


def test_predict_valid(client, booking):
    response = client.post("/predict", json=booking)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["annulation"], bool)
    assert 0 <= body["probabilite_annulation"] <= 1
    assert body["annulation"] == (body["probabilite_annulation"] >= 0.5)


@pytest.mark.parametrize(
    "field, value",
    [
        ("hotel", "Motel"),
        ("adults", -1),
        ("arrival_date_month", "Smarch"),
        ("adr", -10),
        ("country", "france"),
    ],
)
def test_predict_invalid_value(client, booking, field, value):
    booking[field] = value
    assert client.post("/predict", json=booking).status_code == 422


def test_predict_missing_field(client, booking):
    del booking["country"]
    assert client.post("/predict", json=booking).status_code == 422


@pytest.mark.parametrize("hotel", ["City Hotel", "Resort Hotel"])
def test_predict_each_hotel(client, booking, hotel):
    booking["hotel"] = hotel
    response = client.post("/predict", json=booking)
    assert response.status_code == 200
    assert 0 <= response.json()["probabilite_annulation"] <= 1


@pytest.mark.parametrize("row", csv_bookings())
def test_predict_matches_model(client, row):
    response = client.post("/predict", json=row)
    assert response.status_code == 200
    expected = state["model"].predict_proba(pd.DataFrame([row]))[0][1]
    assert response.json()["probabilite_annulation"] == pytest.approx(expected, abs=1e-4)
