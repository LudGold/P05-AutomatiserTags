# StackOverflow Tags API

API FastAPI de prédiction de tags pour les questions StackOverflow.

## Structure
```
stackoverflow-tags-api/
├── app/
│   ├── main.py           # Endpoints FastAPI (/predict, /health)
│   ├── model.py          # Chargement modèles et prédiction
│   └── preprocessing.py  # Pipeline de nettoyage du texte
│
├── models/               # Modèles ML (téléchargés depuis HuggingFace)
│   ├── w2v_classifier.pkl
│   ├── mlb.pkl
│   └── word2vec.bin
│
├── tests/                # Tests unitaires (pytest)
│   └── test_api.py
│
├── streamlit_app/        # Interface de démonstration
│   └── app.py
│
├── requirements.txt      # Dépendances Python
└── README.md             

## Lancer en local
```bash
cd stackoverflow-tags-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API disponible sur **http://127.0.0.1:8000/docs**
- Documentation Swagger : **http://127.0.0.1:8000/docs**
- Health check : **http://127.0.0.1:8000/health**

## 3. Lancer l'interface Streamlit

Dans un **nouveau terminal** (garder l'API active) :
```bash
streamlit run streamlit_app/app.py
```
L'interface s'ouvre sur : **http://localhost:8501**

## Endpoints

- `GET /` : Message de bienvenue
- `GET /health` : Health check
- `POST /predict` : Prédiction de tags

### Endpoint POST /predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "How to read a CSV file with pandas?", "top_k": 5}'

```json

{
  "tags": ["python", "pandas", "csv", "dataframe", "file"],
  "confidence_scores": [0.92, 0.78, 0.65, 0.41, 0.38]
}
```
## Lancer les Tests
```bash
cd stackoverflow-tags-api
py -m unittest tests.test_api -v
```

## Déploiement sur Render

L'API est déployée sur Render : **https://p05-automatisertags.onrender.com/docs**

Modèles : Hébergés sur HuggingFace Hub


## Dépendances principales

FastAPI + Uvicorn (API)
Gensim (Word2Vec)
Scikit-learn (Classification)
NLTK + BeautifulSoup (Prétraitement)
Streamlit (Interface)