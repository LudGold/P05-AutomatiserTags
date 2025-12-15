from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict_tags, load_models

app = FastAPI(
    title="StackOverflow Tags Predictor",
    description="API de suggestion de tags pour les questions StackOverflow",
    version="1.0.0"
)

# Charger les modèles au démarrage (PAS à l'import)
@app.on_event("startup")
def startup_event():
    load_models()


class Question(BaseModel):
    text: str
    top_k: int = 5


class PredictionResponse(BaseModel):
    question: str
    predicted_tags: list
    confidence_scores: list


@app.get("/")
def root():
    return {"message": "API de prédiction de tags StackOverflow", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(question: Question):
    if not question.text.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    tags, scores = predict_tags(question.text, top_k=question.top_k)

    return PredictionResponse(
        question=question.text,
        predicted_tags=tags,
        confidence_scores=scores
    )
