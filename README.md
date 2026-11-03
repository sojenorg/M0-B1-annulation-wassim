# Escale Atlantique — API de prédiction d'annulation

API REST (FastAPI) qui expose un modèle scikit-learn estimant la probabilité qu'une réservation d'hôtel soit annulée. Brief **M0-B1** (FastIA).

## Contexte

Escale Atlantique (un hôtel urbain et un resort) subit environ une annulation pour trois réservations. Son équipe data a entraîné un modèle de classification ; ce service le rend accessible via HTTP.

## Prérequis

- Python 3.11 ou 3.12
- Docker (optionnel, pour le lancement conteneurisé)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Les versions sont figées dans `requirements.txt`. scikit-learn doit rester en 1.6.1 (version d'entraînement du modèle), sinon le fichier `.joblib` ne se charge pas.

## Lancement en local

```bash
uvicorn app.main:app --reload
```

- Santé du service : <http://localhost:8000/health>
- Documentation interactive (Swagger) : <http://localhost:8000/docs>

Si le port 8000 est occupé, ajouter `--port 8001`.

## Lancement avec Docker

```bash
docker build -t fastia-annulation:dev .
docker run --rm -p 8001:8001 fastia-annulation:dev
```

Le conteneur écoute sur le port **8001** : <http://localhost:8001/docs>

Un `HEALTHCHECK` interroge `/health` toutes les 30 s ; `docker ps` affiche `(healthy)` une fois le modèle chargé.

### API + interface web avec Docker Compose

```bash
docker compose up --build
```

Deux conteneurs démarrent : `api` (port 8001) et `ui` (port 8501). L'interface attend que l'API soit
`healthy` avant de démarrer et la joint via `http://api:8001` sur le réseau Compose.

- API : <http://localhost:8001/docs>
- Interface : <http://localhost:8501>

Arrêt : `docker compose down`.

## Tests

```bash
pytest -v
```

16 tests : santé du service, cas valide, entrées invalides (422), les deux types d'hôtel, et fidélité au modèle sur des lignes réelles du CSV.

## Routes

| Méthode | Route | Corps | Réponse |
| --- | --- | --- | --- |
| GET | `/health` | — | `{"status": "ok", "model_loaded": true}` |
| GET | `/info` | — | nom, version, date d'entraînement et colonnes du modèle |
| POST | `/predict` | une réservation | `{"annulation": bool, "probabilite_annulation": float}` |
| POST | `/predict/batch` | liste de réservations | liste de prédictions, même ordre |

## Exemple d'appel à `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/exemple_reservation.json
```

Corps envoyé (`data/exemple_reservation.json`) :

```json
{
  "hotel": "Resort Hotel",
  "lead_time": 164,
  "arrival_date_month": "November",
  "stays_in_weekend_nights": 2,
  "stays_in_week_nights": 5,
  "adults": 2,
  "children": 0,
  "market_segment": "Online TA",
  "deposit_type": "No Deposit",
  "customer_type": "Transient",
  "country": "FRA",
  "adr": 61.2,
  "previous_cancellations": 0,
  "total_of_special_requests": 0,
  "required_car_parking_spaces": 1
}
```

Réponse :

```json
{"annulation": false, "probabilite_annulation": 0.0003}
```

Une entrée invalide (hôtel inconnu, nombre négatif, champ manquant) est refusée avec un code **422** avant d'atteindre le modèle.

Valeurs acceptées pour les champs catégoriels :

| Champ | Valeurs |
| --- | --- |
| `hotel` | `City Hotel`, `Resort Hotel` |
| `arrival_date_month` | nom du mois en anglais (`January` … `December`) |
| `market_segment` | `Aviation`, `Complementary`, `Corporate`, `Direct`, `Groups`, `Offline TA/TO`, `Online TA` |
| `deposit_type` | `No Deposit`, `Non Refund`, `Refundable` |
| `customer_type` | `Contract`, `Group`, `Transient`, `Transient-Party` |
| `country` | code pays en majuscules, 2 ou 3 lettres (`FRA`, `PRT`…) |

## Interface web

Un formulaire HTML minimal (FastAPI + httpx) qui appelle l'API et affiche le résultat.
Avec Docker, voir la section Compose ci-dessus. En local, lancer l'API d'abord, puis dans un second terminal :

```bash
API_URL=http://localhost:8000 uvicorn ui.main:app --port 8501
```

Ouvrir http://localhost:8501 : le formulaire est pré-rempli avec l'exemple, les listes déroulantes
n'offrent que les valeurs acceptées, et une entrée refusée (422) affiche les erreurs renvoyées par l'API.

## Logs

Chaque appel à `/predict` est journalisé dans `logs/api.log` (entrée, prédiction, durée en ms). Rotation à 5 Mo, rétention 7 jours, anciens fichiers compressés en zip. Le dossier `logs/` est ignoré par git.

## Structure

```
app/
  main.py          # application FastAPI : lifespan, logs, routes
  schemas.py       # schémas Pydantic (entrées contraintes, sorties)
model/
  cancellation_model.joblib   # modèle livré (pipeline scikit-learn) — ne pas modifier
  model_info.json             # fiche d'identité du modèle
  train_baseline.py           # script d'entraînement, pour information
data/
  hotel_bookings_sample.csv   # extrait du jeu d'entraînement (1 000 réservations)
  exemple_reservation.json    # exemple de corps de requête pour /predict
ui/
  main.py          # interface web : formulaire qui appelle l'API via httpx
  requirements.txt # dépendances de l'interface seule
tests/
  conftest.py      # fixture `client` (TestClient)
  test_health.py   # tests de /health et de la doc OpenAPI
  test_predict.py  # tests de /predict
Dockerfile         # image de l'API
Dockerfile.ui      # image de l'interface
docker-compose.yml # API + interface
.dockerignore
pytest.ini
requirements.txt
```
