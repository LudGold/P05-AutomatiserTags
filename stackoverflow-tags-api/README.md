# StackOverflow Tags API

API FastAPI de prédiction de tags pour les questions StackOverflow.

## Structure
```
app/
├── main.py           # Endpoints FastAPI (/predict, /health)
├── model.py          # Chargement modèles et prédiction
└── preprocessing.py  # Pipeline de nettoyage du texte

models/               # Modèles ML (téléchargés depuis HuggingFace)
├── w2v_classifier.pkl
├── mlb.pkl
└── word2vec.bin

tests/                # Tests unitaires
streamlit_app/        # Interface de démonstration
```

## Lancer en local
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API disponible sur http://127.0.0.1:8000

## Endpoints

- `GET /` : Message de bienvenue
- `GET /health` : Health check
- `POST /predict` : Prédiction de tags

## Déploiement

L'API est déployée sur Render : https://stackoverflow-tags-api.onrender.com