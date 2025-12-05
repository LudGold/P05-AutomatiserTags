import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAPI(unittest.TestCase):
    """Tests unitaires pour l'API de prédiction de tags."""

    @classmethod
    def setUpClass(cls):
        """Initialise le client de test une seule fois."""
        cls.client = TestClient(app)

    def test_root(self):
        """Test de l'endpoint racine."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")

    def test_health(self):
        """Test de l'endpoint health check."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_predict_success(self):
        """Test d'une prédiction réussie."""
        response = self.client.post(
            "/predict",
            json={"text": "How do I read a CSV file in Python using pandas?", "top_k": 5}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Vérifie la structure de la réponse
        self.assertIn("question", data)
        self.assertIn("predicted_tags", data)
        self.assertIn("confidence_scores", data)
        
        # Vérifie qu'on a bien 5 tags
        self.assertEqual(len(data["predicted_tags"]), 5)
        self.assertEqual(len(data["confidence_scores"]), 5)
        
        # Vérifie que "python" est dans les tags prédits
        self.assertIn("python", data["predicted_tags"])

    def test_predict_empty_text(self):
        """Test avec un texte vide."""
        response = self.client.post(
            "/predict",
            json={"text": "", "top_k": 5}
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_different_top_k(self):
        """Test avec un nombre différent de tags."""
        response = self.client.post(
            "/predict",
            json={"text": "JavaScript async await promises", "top_k": 3}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["predicted_tags"]), 3)


if __name__ == "__main__":
    unittest.main()