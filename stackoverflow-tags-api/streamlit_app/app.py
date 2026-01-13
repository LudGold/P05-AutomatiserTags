import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="StackOverflow Tags Predictor",
    layout="centered"
)

# URL de l'API (local)
API_URL = st.sidebar.text_input(
    "URL de l'API",
    value="https://p05-automatisertags.onrender.com"
)


st.title(" StackOverflow Tags Predictor")
st.markdown("Entrez une question technique et obtenez des suggestions de tags.")

# Zone de saisie
question = st.text_area(
    "Votre question :",
    placeholder="Ex: How do I read a CSV file in Python using pandas?",
    height=100
   
    
)

# Nombre de tags
top_k = st.slider("Nombre de tags à suggérer :", min_value=1, max_value=10, value=5)

# Bouton de prédiction
if st.button(" Prédire les tags", type="primary"):
    if question.strip():
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": question, "top_k": top_k},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Tags prédits :")
                    
                    # Affichage des tags avec scores
                    for tag, score in zip(data["predicted_tags"], data["confidence_scores"]):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**`{tag}`**")
                        with col2:
                            st.progress(score)
                            st.caption(f"{score:.1%}")
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Impossible de se connecter à l'API. Vérifiez qu'elle est lancée.")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
    else:
        st.warning("Posez votre question.")

# Footer
st.markdown("---")
st.caption("Projet StackOverflow Tags - MLOps - LG 2025")