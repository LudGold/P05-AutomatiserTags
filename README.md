# StackOverflow Tags Predictor

API de suggestion automatique de tags pour les questions StackOverflow.

## Structure du projet
```
stackoverflow-tags-api/
├── app/                    # Code de l'API FastAPI
│   ├── main.py            # Endpoints API
│   ├── model.py           # Chargement modèle et prédiction
│   └── preprocessing.py   # Prétraitement du texte
├── models/                 # Modèles sauvegardés (non versionnés)
│   ├── w2v_classifier.pkl
│   ├── mlb.pkl
│   └── word2vec.model
├── tests/                  # Tests unitaires
├── streamlit_app/          # Interface de démo
└── .github/workflows/      # CI/CD GitHub Actions
```

## Installation
```bash
cd stackoverflow-tags-api
pip install -r requirements.txt
```

## 1. Lancer MLflow UI

Pour visualiser les expérimentations :
```bash
cd P05_Projet_tags_StackOverflow
mlflow ui
```

Puis ouvrir : **http://127.0.0.1:5000**

## 2. Lancer l'API
```bash
cd stackoverflow-tags-api
py -m uvicorn app.main:app --reload
```

L'API tourne sur : **http://127.0.0.1:8000**

- Documentation Swagger : **http://127.0.0.1:8000/docs**
- Health check : **http://127.0.0.1:8000/health**

## 3. Lancer l'interface Streamlit

Dans un **nouveau terminal** (garder l'API active) :
```bash
cd stackoverflow-tags-api
streamlit run streamlit_app/app.py
```

L'interface s'ouvre sur : **http://localhost:8501**

## 4. Lancer les tests
```bash
“Word2Vec vectors loaded from binary format to ensure CI compatibility”
cd stackoverflow-tags-api
py -m unittest tests.test_api -v
```

## Utilisation de l'API

### Endpoint POST /predict
```json
{
  "text": "How do I read a CSV file in Python using pandas?",
  "top_k": 5
}
```

### Réponse
```json
{
  "question": "How do I read a CSV file in Python using pandas?",
  "predicted_tags": ["python", "csv", "pandas", "dataframe", "file"],
  "confidence_scores": [0.99, 0.85, 0.72, 0.65, 0.52]
}
```

## Résultats des modèles

| Modèle   | F1_Micro | TCTP_Top5 | Temps (sec) |
|----------|----------|-----------|-------------|
| Word2Vec | 0.308    | 0.654     | 237         |
| BERT     | 0.261    | 0.580     | 2169        |
| USE      | 0.259    | 0.629     | 572         |
| TF-IDF   | 0.107    | 0.506     | 92          |

**Modèle déployé : Word2Vec** (meilleur compromis performance/temps)
