import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Télécharger les ressources NLTK nécessaires (une seule fois)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Stop words
STOP_WORDS = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']


def clean_html_and_code(text):
    """Supprime les balises HTML et extrait le texte brut."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text(separator=' ')
    cleaned_text = re.sub('\n', ' ', cleaned_text)
    return cleaned_text


def tokenizer_fct(sentence):
    """Tokenize en remplaçant les caractères spéciaux."""
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens


def stop_word_filter_fct(list_words):
    """Filtre les stopwords et mots courts."""
    filtered_w = [w for w in list_words if w not in STOP_WORDS]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2


def lower_start_fct(list_words):
    """Lowercase et filtre les mentions/URLs."""
    lw = [w.lower() for w in list_words 
          if (not w.startswith("@")) and (not w.startswith("http"))]
    return lw


def lemma_fct(list_words):
    """Lemmatization des mots."""
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w


def preprocess_text(text):
    """
    Pipeline complet de prétraitement pour Word2Vec.
    Reproduit exactement transform_bow_lem_fct utilisé à l'entraînement.
    """
    # 1. Nettoyage HTML
    cleaned = clean_html_and_code(text)
    
    # 2. Tokenization
    tokens = tokenizer_fct(cleaned)
    
    # 3. Stopwords
    tokens = stop_word_filter_fct(tokens)
    
    # 4. Lowercase + filtrage
    tokens = lower_start_fct(tokens)
    
    # 5. Lemmatization
    tokens = lemma_fct(tokens)
    
    return tokens


def preprocess_for_vectorization(text):
    """Retourne le texte prétraité sous forme de string (pour compatibilité)."""
    tokens = preprocess_text(text)
    return ' '.join(tokens)