# StackOverflow Tags Predictor

Projet de Machine Learning pour suggérer automatiquement des tags aux questions StackOverflow.
Ce projet compare 4 approches (TF-IDF, Word2Vec, BERT, USE) et déploie le meilleur modèle (Word2Vec) via une API FastAPI sur le cloud
L'objectif étant de Développer un algorithme de Machine Learning qui suggère automatiquement les 5 tags les plus pertinents pour une question technique, afin d'aider les nouveaux utilisateurs de StackOverflow

## Structure du projet

### Structure du projet

```text
P05_Projet_tags_StackOverflow/
├── notebooks/                              # Exploration et prétraitement
│   ├── 01_analyse_exploratoire.ipynb
│   ├── 02_requete_api_stackoverflow.ipynb
│   ├── 03_approche_non_supervisee_LDA.ipynb
│   └── 04_approche_supervisee_mlflow.ipynb
├── stackoverflow-tags-api/                 # API de prédiction (déployée)
│   ├── app/                                # Code FastAPI
│   ├── models/                             # Modèles ML sérialisés
│   ├── tests/                              # Tests unitaires
│   ├── streamlit_app/                      # Interface de démonstration
│   ├── mlruns/                             # Tracking MLflow
│   ├── requirements.txt                    # Dépendances API
│   └── README.md                           # Doc API
├── data/                                   # Données (CSV)
├── Note_Technique_MLOps.docx               # Étude MLOps (Kedro, EvidentlyAI...)
├── Presentation_P05.pptx                   # Support de soutenance
├── requirements.txt                        # Dépendances globales
└── README.md
## Liens importants 

API déployée https://p05-automatisertags.onrender.com/docs
GitHub https://github.com/LudGold/P05-AutomatiserTags 
Modèles sur HuggingFace https://huggingface.co/LudGold/stackoverflow-tags-models

## Résultats des modèles

| Modèle   | F1_Micro | TCTP_Top5 | Temps (sec) |
|----------|----------|-----------|-------------|
| Word2Vec | 0.308    | 0.654     | 237         |
| BERT     | 0.261    | 0.580     | 2169        |
| USE      | 0.259    | 0.629     | 572         |
| TF-IDF   | 0.107    | 0.506     | 92          |

**Modèle déployé : Word2Vec + Logistic Regression** (meilleur compromis performance/temps)

## Installation pour les notebooks
pip install -r requirements.txt

## 1. Lancer MLflow UI

Pour visualiser les expérimentations :
```bash
cd P05_Projet_tags_StackOverflow
mlflow ui
```
Puis ouvrir : **http://127.0.0.1:5000**

## 2. Lancer l'API en local
```bash
cd stackoverflow-tags-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```
