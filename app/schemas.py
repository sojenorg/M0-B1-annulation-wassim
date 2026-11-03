"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Literal

from pydantic import BaseModel, Field

Hotel = Literal["City Hotel", "Resort Hotel"]
Month = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MarketSegment = Literal[
    "Aviation", "Complementary", "Corporate", "Direct",
    "Groups", "Offline TA/TO", "Online TA",
]
DepositType = Literal["No Deposit", "Non Refund", "Refundable"]
CustomerType = Literal["Contract", "Group", "Transient", "Transient-Party"]


class Booking(BaseModel):
    """Une réservation, telle que saisie dans le logiciel de gestion hôtelière.

    Les noms de champs sont ceux du jeu de données d'entraînement :
    le modèle attend exactement ces colonnes.
    """

    hotel: Hotel
    lead_time: int = Field(ge=0, le=1000)
    arrival_date_month: Month
    stays_in_weekend_nights: int = Field(ge=0, le=30)
    stays_in_week_nights: int = Field(ge=0, le=60)
    adults: int = Field(ge=0, le=10)
    children: int = Field(ge=0, le=10)
    market_segment: MarketSegment
    deposit_type: DepositType
    customer_type: CustomerType
    country: str = Field(min_length=2, max_length=3, pattern=r"^[A-Z]+$")
    adr: float = Field(ge=0, le=10000)
    previous_cancellations: int = Field(ge=0, le=100)
    total_of_special_requests: int = Field(ge=0, le=10)
    required_car_parking_spaces: int = Field(ge=0, le=10)


class PredictionResponse(BaseModel):
    """Réponse de /predict."""

    annulation: bool
    probabilite_annulation: float = Field(ge=0, le=1)


class ModelInfo(BaseModel):
    """Réponse de /info."""

    model_name: str
    version: str
    trained_on: str
    features: list[str]
