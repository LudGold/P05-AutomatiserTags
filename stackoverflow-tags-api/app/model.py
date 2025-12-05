import numpy as np
import joblib
from gensim.models import Word2Vec
from app.preprocessing import preprocess_text

# Chemins des modèles
CLASSIFIER_PATH = "models/w2v_classifier.pkl"
MLB_PATH = "models/mlb.pkl"
W2V_PATH = "models/word2vec.model"

# Variables globales pour les modèles (chargés une seule fois)
classifier = None
mlb = None
w2v_model = None


def load_models():
    """Charge les modèles en mémoire."""
    global classifier, mlb, w2v_model
    
    if classifier is None:
        print("Chargement des modèles...")
        classifier = joblib.load(CLASSIFIER_PATH)
        mlb = joblib.load(MLB_PATH)
        w2v_model = Word2Vec.load(W2V_PATH)
        print(" Modèles chargés !")
    
    return classifier, mlb, w2v_model


def text_to_vector(tokens, model, vector_size=100):
    """
    Convertit une liste de tokens en vecteur moyen Word2Vec.
    Reproduit exactement la vectorisation utilisée à l'entraînement.
    """
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) == 0:
        # Si aucun mot n'est dans le vocabulaire, retourne un vecteur nul
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)


def predict_tags(text, top_k=5):
    """cd 
    Prédit les tags pour une question donnée.
    
    Args:
        text: Le texte de la question (titre + body)
        top_k: Nombre de tags à retourner
    
    Returns:
        tags: Liste des tags prédits
        scores: Liste des scores de confiance
    """
    # Charger les modèles si nécessaire
    clf, mlb, w2v = load_models()
    
    # 1. Prétraitement
    tokens = preprocess_text(text)
    
    # 2. Vectorisation Word2Vec
    vector = text_to_vector(tokens, w2v, vector_size=100)
    vector = vector.reshape(1, -1)  # Shape (1, 100) pour la prédiction
    
    # 3. Prédiction des probabilités
    probas = clf.predict_proba(vector)[0]
    
    # 4. Récupérer les top_k tags avec les meilleurs scores
    top_indices = np.argsort(probas)[::-1][:top_k]
    
    tags = [mlb.classes_[i] for i in top_indices]
    scores = [round(float(probas[i]), 4) for i in top_indices]
    
    return tags, scores