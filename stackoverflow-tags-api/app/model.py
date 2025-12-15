import os
import numpy as np
import joblib
from gensim.models import Word2Vec
from huggingface_hub import hf_hub_download
from app.preprocessing import preprocess_text

HF_REPO = "LudGold/stackoverflow-tags-models"
HF_REVISION = os.getenv("HF_REVISION", "main")
HF_TOKEN = os.getenv("HF_TOKEN")

MODELS_DIR = "models"

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "w2v_classifier.pkl")
MLB_PATH = os.path.join(MODELS_DIR, "mlb.pkl")
W2V_PATH = os.path.join(MODELS_DIR, "word2vec.model")

classifier = None
mlb = None
w2v_model = None


def _ensure_file(local_path: str, filename: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(local_path):
        return local_path

    # Télécharge depuis HF et retourne le chemin exact
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,            # tes fichiers sont à la racine du repo
        revision=HF_REVISION,
        token=HF_TOKEN,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )


def load_models():
    global classifier, mlb, w2v_model

    if classifier is None or mlb is None or w2v_model is None:
        clf_path = _ensure_file(CLASSIFIER_PATH, "w2v_classifier.pkl")
        mlb_path = _ensure_file(MLB_PATH, "mlb.pkl")
        w2v_path = _ensure_file(W2V_PATH, "word2vec.model")

        classifier = joblib.load(clf_path)
        mlb = joblib.load(mlb_path)
        w2v_model = Word2Vec.load(w2v_path)

    return classifier, mlb, w2v_model
